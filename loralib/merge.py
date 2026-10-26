"""Building blocks of Compress then Merge (CtM; He et al., ICML 2026).

CtM merges several LoRAs of one layer into a single low-rank layer: it chooses the
rank-k subspaces first, then merges the LoRAs' coordinates in those subspaces. Here the
recipe chooses k from `tucker_spectrum`, as it chooses the rank of a single LoRA from
its singular values.
"""

import torch


def core_space(ups, downs):
    """Shared orthonormal bases of the LoRAs and each LoRA's lossless core.

    ups[t] is (m, r_t) and downs[t] is (r_t, n). Returns Q_up (m, a), Q_down (n, b) and
    cores (T, a, b) with ups[t] @ downs[t] == Q_up @ cores[t] @ Q_down.mT, where a and b
    are at most the sum of the ranks.
    """
    Q_up, R_up = torch.linalg.qr(torch.cat(ups, dim=1))
    Q_down, R_down = torch.linalg.qr(torch.cat(downs, dim=0).mT)
    ranks = [down.shape[0] for down in downs]
    cores = torch.stack(
        [
            Ru @ Rd.mT
            for Ru, Rd in zip(R_up.split(ranks, dim=1), R_down.split(ranks, dim=1))
        ]
    )
    return Q_up, Q_down, cores


def _unfold(X):
    """(T, a, b) -> (a, T*b): the matrices side by side."""
    return X.transpose(0, 1).reshape(X.shape[1], -1)


def _top_left_singular(X, k):
    """The k top left singular vectors of X, as orthonormal columns.

    They are the top eigenvectors of X @ X.T, which also complete them with an
    orthonormal basis when k exceeds X's rank. For the small float64 matrices here,
    eigh is faster than an SVD and accurate enough.
    """
    return torch.linalg.eigh(X @ X.mT)[1][:, -k:].flip(1)


def tucker_spectrum(cores):
    """Higher-order SVD directions of a (T, a, b) stack, and the norm of each component.

    Returns P_up (a, a), P_down (b, b), s (n,) sorted descending with n = min(a, b),
    and up_cols, down_cols (n,): the columns of the two directions of each component.
    Component j adds output direction j + a - n and input direction j + b - n, so the
    larger side starts with |a - b| directions of its own: s_j^2 = E(j) - E(j-1) with
    E(j) = sum_t ||P_up[:, :j + a - n].T @ cores[t] @ P_down[:, :j + b - n]||_F^2.
    The s_j^2 add up to the energy of the stack. For a single matrix, the directions
    and norms are its singular vectors and values.
    """
    a, b = cores.shape[1:]
    n = min(a, b)
    P_up = _top_left_singular(_unfold(cores), a)
    P_down = _top_left_singular(_unfold(cores.mT), b)
    energies = (P_up.T @ cores @ P_down).square().sum(0)
    j = torch.arange(n)
    kept = energies.cumsum(0).cumsum(1)[j + a - n, j + b - n]
    s, order = (
        torch.diff(kept, prepend=kept.new_zeros(1))
        .clamp(min=0)
        .sqrt()
        .sort(descending=True)
    )
    return P_up, P_down, s, order + a - n, order + b - n


def tucker2(cores, ranks, max_iter=1000, tol=1e-12):
    """Orthonormal P_up (a, k_up) and P_down (b, k_down) maximizing the kept energy
    sum_t ||P_up.T @ cores[t] @ P_down||_F^2, for ranks = (k_up, k_down).

    Higher-order orthogonal iteration from the SVD of the unfoldings, which is what
    tensorly's partial_tucker computes for modes [0, 1]. It stops when the kept energy
    grows by less than `tol` of itself. The gain shrinks with the square of the change
    in directions, so `tol` must be small, and float64 cores make it reachable.
    """
    k_up, k_down = ranks
    P_up = _top_left_singular(_unfold(cores), k_up)
    P_down = _top_left_singular(_unfold(cores.mT), k_down)
    energy = (P_up.T @ cores @ P_down).square().sum().item()
    for _ in range(max_iter):
        P_up = _top_left_singular(_unfold(cores @ P_down), k_up)
        P_down = _top_left_singular(_unfold(cores.mT @ P_up), k_down)
        previous, energy = energy, (P_up.T @ cores @ P_down).square().sum().item()
        if energy - previous <= tol * energy:
            break
    return P_up, P_down


def ties(cores, density):
    """TIES-merging (Yadav et al., 2023) of a (T, a, b) stack.

    Keeps the floor(density * d) + 1 largest-magnitude of each matrix's d entries (all
    of them when density is 1), as the original CtM code does, elects each entry's sign
    as the sign of the sum of the kept values, and averages the kept values that have
    that sign.
    """
    flat = cores.flatten(1)
    d = flat.shape[1]
    kth = flat.abs().topk(min(d, int(d * density) + 1), dim=1).values[:, -1:]
    kept = torch.where(flat.abs() >= kth, flat, 0.0)
    sign = kept.sum(0).sign()
    agree = (kept.sign() == sign) & (kept != 0)
    merged = (kept * agree).sum(0) / agree.sum(0).clamp(min=1)
    return merged.view(cores.shape[1:])
