import torch

# The SHOT LRF, as per https://link.springer.com/chapter/10.1007/978-3-642-15558-1_26
def SHOT(p, neigh_p, R):
    # compute covariance matrix cov
    # print(neigh_p)
    sum_wq = 0
    cov = torch.zeros((len(p), len(p)))
    q_m = torch.mean(neigh_p, 0)
    for q in neigh_p:
        t = q - q_m
        wq = R - torch.linalg.vector_norm(t)
        sum_wq += wq
        cov += wq * (t[:, None] @ t[None, :])

    cov /= sum_wq

    # eigenvectors
    _, V = torch.linalg.eigh(cov)
    # V is in ascending order, so we reorder it in descending order
    V = V.T
    V[[0, 2]] = V[[2, 0]]

    #sign disambiguation
    Sx_plus = 0
    Sz_plus = 0

    for q in neigh_p:
        if torch.dot(q - q_m, V[0]) >= 0:
            Sx_plus += 1
        if torch.dot(q - q_m, V[2]) >= 0:
            Sz_plus += 1

    if 2 * Sx_plus < len(neigh_p):
        V[0] *= (-1)
    if 2 * Sz_plus < len(neigh_p):
        V[2] *= (-1)

    V[1] = torch.linalg.cross(V[2], V[0])

    return V.T

# CA method described in https://www.researchgate.net/publication/220659665_On_the_Repeatability_and_Quality_of_Keypoints_for_Local_Feature-based_3D_Object_Retrieval_from_Cluttered_Scenes
def global_lrf(graph):
    n = graph.pos.size(0)
    P = graph.pos
    # Compute covariance matrix
    cov = P.T @ P

    # Compute eigenvectors
    _, V = torch.linalg.eigh(cov)

    V = V.T
    V[[0, 2]] = V[[2, 0]]

    # Sign disambiguation
    dot_products = P @ V[0]
    Sx_plus = torch.sum(dot_products >= 0)
    dot_products = P @ V[1]
    Sy_plus = torch.sum(dot_products >= 0)
    dot_products = P @ V[2]
    Sz_plus = torch.sum(dot_products >= 0)

    if 2 * Sx_plus < n:
        V[0] *= (-1)
    if 2 * Sy_plus < n:
        V[1] *= (-1)
    if 2 * Sz_plus < n:
        V[2] *= (-1)

    return V.T


    # conda remove --name dir_gsn --all
    # conda clean --all --yes