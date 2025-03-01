import torch
import torch.nn.functional as F

def pairwise_distances(x, y=None, metric='cosine'):
    if y is None:
        y = x
    if metric == 'cosine':
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        dist = 1 - torch.mm(x_norm, y_norm.t())
    elif metric == 'euclidean':
        dist = torch.cdist(x, y, p=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return dist


def clust_rank(mat, initial_rank=None, distance='cosine', use_tw_finch=False):
    s = mat.shape[0]
    
    if initial_rank is not None:
        orig_dist = []
        
    if use_tw_finch:
        loc = mat[:, -1]
        mat = mat[:, :-1]            
        loc_dist = torch.sqrt((loc[:, None] - loc[:, None].T)**2).to(mat.device)
        
    else:
        loc_dist = torch.tensor(1., device=mat.device)

    orig_dist = pairwise_distances(mat, mat, metric=distance)
    orig_dist = orig_dist * loc_dist
    orig_dist.fill_diagonal_(1e12)
    initial_rank = torch.argmin(orig_dist, dim=1)

    # The Clustering Equation
    A = torch.zeros((s, s), device=mat.device)
    A[torch.arange(s), initial_rank] = 1
    A = A + torch.eye(s, device=mat.device)
    A = A @ A.T
    A.fill_diagonal_(0)
    
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[orig_dist > min_sim] = 0

    num_clust, u = connected_components(a)
    return u, num_clust


def connected_components(a):
    n = a.shape[0]
    visited = torch.zeros(n, dtype=torch.bool, device=a.device)
    labels = torch.full((n,), -1, dtype=torch.int64, device=a.device)
    label = 0

    def dfs(node):
        stack = [node]
        while stack:
            v = stack.pop()
            for neighbor in torch.nonzero(a[v], as_tuple=False).flatten():
                if not visited[neighbor]:
                    visited[neighbor] = True
                    labels[neighbor] = label
                    stack.append(neighbor)

    for i in range(n):
        if not visited[i]:
            visited[i] = True
            labels[i] = label
            dfs(i)
            label += 1

    return label, labels


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = torch.unique(u, return_counts=True)
    umat = torch.zeros((s, len(un)), device=M.device)
    umat[torch.arange(s), u] = 1
    return (umat.T @ M) / nf[..., None]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = torch.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero(as_tuple=True)
    v = torch.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = torch.zeros_like(adj)
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance, use_tw_finch=False):
    iter_ = len(torch.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance, use_tw_finch=use_tw_finch)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', tw_finch=False, ensure_early_exit=False, verbose=True):
    """ FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param tw_finch: Run TW_FINCH on video data.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquib.sarfraz@kit.edu)
    Karlsruhe Institute of Technology (KIT)
    """
    device = data.device
    # data = torch.tensor(data, device=device)

    if tw_finch:
        n_frames = data.shape[0]
        time_index = (torch.arange(n_frames, device=device) + 1.) / n_frames       
        data = torch.cat([data, time_index[..., None]], axis=1)
        verbose = False

    # Cast input data to float32
    data = data.float()
    
    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance=distance, use_tw_finch=tw_finch)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = torch.max(orig_dist * adj)

    exit_clust = 5
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:        
        adj, orig_dist = clust_rank(mat, initial_rank, distance=distance, use_tw_finch=tw_finch)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = torch.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance, use_tw_finch=tw_finch)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c

