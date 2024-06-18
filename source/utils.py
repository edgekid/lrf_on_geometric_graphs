import time
import torch.nn.functional as F
import torch
import numpy as np

from scipy.stats import ortho_group
from mpnn import MPNNModel, CoordMPNNModel

from lrf_methods import SHOT, global_lrf
from torch_geometric.loader import DataLoader

from tqdm import tqdm

def train(model, dataloader, loss_fn, opt, device):
    """
    Perform a single training epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.DataLoader): The dataloader containing the training data.
        loss_fn (Callable): The loss function to calculate the loss.
        opt (torch.optim.Optimizer): The optimizer to update the model parameters.

    Returns:
        float: Mean loss value.
        float: Mean accuracy value.

    """
    model.train()
    losses = []

    for data in dataloader:
        opt.zero_grad()  # zero gradients in the optimizer
        data = data.to(device)

        output = model(data)  # forward pass through the model
        loss = loss_fn(output, data.y)  # calculate the loss

        loss.backward()  # backpropagate the loss
        opt.step()  # update the model parameters

        losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss

# decorator to disable gradient calculation during evaluation
@torch.no_grad()
def eval(model, dataloader, loss_fn, device):
    """
    Evaluate the model on the given dataloader.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader containing the evaluation data.
        loss_fn (Callable): The loss function to calculate the loss.
        device
    Returns:
        float: Mean loss value.
    """
    model.eval()

    losses = []
    for data in dataloader:
        data = data.to(device)

        output = model(data)  # forward pass through the model
        loss = loss_fn(output, data.y)  # calculate the loss

        losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss


def run_experiment(model, model_name, train_loader, val_loader, test_loader, n_epochs=100):
    print(f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")

    if torch.cuda.is_available():
        print("Using GPU, device name:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print("No GPU found, using CPU instead.")
        device = torch.device("cpu")

    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    # Adam optimizer with LR 1e-3
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = F.torch.nn.L1Loss()

    # LR scheduler which decays LR when validation metric doesn't improve
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.9, patience=5, min_lr=0.00001)

    print("\nStart training:")
    best_val_loss = None
    perf_per_epoch = [] # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()

    for i in range(1, n_epochs + 1):
        # train the model for one epoch
        train_loss = train(model, train_loader, loss_fn, opt, device)

        # evaluate the model on the test data
        val_loss = eval(model, val_loader, loss_fn, device)

        # check if the current test accuracy is better than the best test accuracy so far
        if best_val_loss is None or val_loss <= best_val_loss:
            # Evaluate model on test set if validation metric improves
            test_loss = eval(model, test_loader, loss_fn, device)
            best_val_loss = val_loss

        # print the training and test metrics for the current epoch
        if i % 1 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {i:03d}, Loss: {train_loss:.7f}, '
                  f'Val MAE: {val_loss:.7f}, Test MAE: {test_loss:.7f}')

    t = time.time() - t
    train_time = t/60
    print(f"\nDone! Training took {train_time:.2f} mins. Best validation MAE: {best_val_loss:.7f}, corresponding test MAE: {test_loss:.7f}.")

    return best_val_loss, test_loss, train_time, perf_per_epoch

def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q


def rot_trans_invariance_unit_test(dataset):
    """Unit test for checking whether a module (GNN model/layer) is
    rotation and translation invariant.
    """
    
    model = CoordMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    it = iter(dataloader)
    invariant = 0

    num_tests = len(dataset)
    for i in tqdm(range(num_tests)):
        graph = next(it)

        # Forward pass on original example
        # Note: We have written a conditional forward pass so that the same unit
        #       test can be used for both the GNN model as well as the layer.
        #       The functionality for layers will be useful subsequently.

        temp = graph.pos.clone()
        lrf_projection_SHOT(graph)
        lrf1 = graph.pos.clone()

        if isinstance(model, MPNNModel):
            out_1 = model(graph)
        else: # if ininstance(module, MessagePassing):
            out_1 = model(graph.x, graph.pos, graph.edge_index, graph.edge_attr)

        Q = random_orthogonal_matrix(dim=3)
        t = torch.rand(3)
        
        graph.pos = temp
        # Perform random rotation + translation on data.
        graph.pos = graph.pos @ Q.T + t.repeat(graph.pos.size(0), 1)
        
        lrf_projection_SHOT(graph)
        lrf2 = graph.pos.clone()

        # Forward pass on rotated + translated example
        if isinstance(model, MPNNModel):
            out_2 = model(graph)
        else: # if ininstance(module, MessagePassing):
            out_2 = model(graph.x, graph.pos, graph.edge_index, graph.edge_attr)

        # Check whether output varies after applying transformations.
        # Since we are comparing floating points, we need to check
        # if the two numbers are within some small epsilon of each other
        #

        invariant += torch.allclose(out_1, out_2, atol=1e-04)
    return invariant / num_tests * 100

def lrf_projection_SHOT(graph):
    R = 100 # 1 = mask is mostly False (underfit); 3 = a few are False; 5 = mask is moslty True
    n = graph.x.size(0)
    dists = (graph.pos[:, None] - graph.pos[None, :]).float()
    n_mask = torch.linalg.vector_norm(dists, dim = -1) <= R
    
    pos = graph.pos.clone()

    for x in range(n):
        neigh = pos[n_mask[x]].clone()
        mean = torch.mean(neigh, 0)
        for v in neigh:
            v -= mean
        LRF = SHOT(pos[x] - mean, neigh, R)
        
        graph.pos[x] = LRF.T @ (pos[x] - mean)
    
    return graph

def lrf_projection_global(graph):
    # Standardize the points
    mean = torch.mean(graph.pos, 0)
    graph.pos = graph.pos - mean
    
    # Compute LRF
    LRF = global_lrf(graph)
    graph.pos = graph.pos @ LRF # we do x = LRF^(-1) @ x for each point x

    return graph