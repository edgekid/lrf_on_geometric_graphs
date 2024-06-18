from torch_geometric.datasets import QM9
import torch
from torch_geometric.utils import remove_self_loops
from utils import lrf_projection_global, lrf_projection_SHOT

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from utils import rot_trans_invariance_unit_test

class Dataset():
    def __init__(self, lrf_method = '', target = 0):
        path = './qm9'
        self.data_size = 10000
        self.train_size = self.data_size * 8 // 10
        self.test_size = self.train_size  // 10
        self.valid_size = self.data_size - self.train_size - self.test_size
        # Load the QM9 dataset with the transforms defined

        dataset = QM9(path).shuffle()[:self.data_size]
        
        # Perform invariance test on global method
        # print(f"Invariance for global method: {rot_trans_invariance_unit_test(dataset)}%.")
        
        self.dset = [self.transform_data(x, target, lrf_method) for x in tqdm(dataset)]

        
    def transform_data(self, data, target, lrf_method):
        # Modify the labels vector per data sample to only keep the label for a specific target (there are 19 targets in QM9).
        data.y = data.y[:, target]

        # Fully-connect the graphs
        data = self.complete_graph(data)

        data = self.project_onto_lrf(data, lrf_method)

        return data

    """
    Adds all pairwise edges into the edge index per data sample, then removes self loops, i.e. it builds a fully connected or complete graph
    """
    def complete_graph(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data
    
    """
    Projects the node positional vector onto the graph's LRF for each data sample.
    """
    def project_onto_lrf(self, data, lrf_method):
        if lrf_method == "Global":
            return lrf_projection_global(data)
        elif lrf_method == "SHOT":
            return lrf_projection_SHOT(data)
        return data
    
    def split_data(self):
        train_dataset = self.dset[:self.train_size]
        val_dataset = self.dset[self.train_size:(self.train_size + self.valid_size)]
        test_dataset = self.dset[(self.train_size + self.valid_size):self.data_size]
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        return train_loader, val_loader, test_loader