import torch


def fast_decompose(up, down):
    Qu, Ru = torch.linalg.qr(up)
    Qd, Rd = torch.linalg.qr(down.T)
    Uc, Sc, Vhc = torch.linalg.svd(Ru @ Rd.T)
    return Qu @ Uc, Sc, Vhc @ Qd.T


def load_lora_layer(lora_file, name, **to_kwargs):
    alpha = lora_file.get_tensor(f"{name}.alpha").item()
    down = lora_file.get_tensor(f"{name}.lora_down.weight").to(**to_kwargs)
    up = lora_file.get_tensor(f"{name}.lora_up.weight").to(**to_kwargs)
    return alpha, up, down


def outer_cosine_sim(U1, U2):
    U1n = U1 / torch.linalg.norm(U1, dim=0)
    U2n = U2 / torch.linalg.norm(U2, dim=0)
    return U1n.T @ U2n


# adapted from scipy.stats.special_ortho_group, licensed under BSD 3-clause
# https://github.com/philipdeboer/scipy/blob/4c08f6a/scipy/stats/_multivariate.py#L3077
def special_ortho_group(dim, size: int = 1, generator=None):
    r"""
    Draw random samples from SO(N).

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).

    The `dim` keyword specifies the dimension N.

    Parameters
    ----------
    dim : integer
        Dimension of rotation space (N).
    size : integer, optional
        Number of samples to draw (default 1).

    Notes
    ----------
    This function is based on the random_rot code from the MDP Toolkit,
    https://github.com/mdp-toolkit/mdp-toolkit

    Return a random rotation matrix, drawn from the Haar distribution
    (the only uniform distribution on SO(n)).
    The algorithm is described in the paper
    Stewart, G.W., "The efficient generation of random orthogonal
    matrices with an application to condition estimators", SIAM Journal
    on Numerical Analysis, 17(3), pp. 403-409, 1980.
    For more information see
    http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization

    Examples
    --------
    >>> from scipy.stats import special_ortho_group
    >>> x = special_ortho_group.rvs(3)

    >>> np.dot(x, x.T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    >>> import scipy.linalg
    >>> scipy.linalg.det(x)
    1.0

    This generates one random matrix from SO(3). It is orthogonal and
    has a determinant of 1.

    """
    size = int(size)
    size = (size,) if size > 1 else ()

    # H represents a (dim, dim) matrix, while D represents the diagonal of
    # a (dim, dim) diagonal matrix. The algorithm that follows is
    # broadcasted on the leading shape in `size` to vectorize along
    # samples.
    H = torch.empty(*size, dim, dim)
    H[..., :, :] = torch.eye(dim)
    D = torch.empty(*size, dim)

    for n in range(dim - 1):
        # x is a vector with length dim-n, xrow and xcol are views of it as
        # a row vector and column vector respectively. It's important they
        # are views and not copies because we are going to modify x
        # in-place.
        x = torch.randn(*size, dim - n, generator=generator)
        xrow = x[..., None, :]
        xcol = x[..., :, None]

        # This is the squared norm of x, without vectorization it would be
        # dot(x, x), to have proper broadcasting we use matmul and squeeze
        # out (convert to scalar) the resulting 1x1 matrix
        norm2 = torch.matmul(xrow, xcol).squeeze((-2, -1))

        x0 = x[..., 0].clone()
        D[..., n] = torch.where(x0 != 0, torch.sign(x0), 1)
        x[..., 0] += D[..., n] * torch.sqrt(norm2)

        # In renormalizing x we have to append an additional axis with
        # [..., None] to broadcast the scalar against the vector x
        x /= torch.sqrt((norm2 - x0.square_() + x[..., 0].square()) * 0.5)[..., None]

        # Householder transformation, without vectorization the RHS can be
        # written as outer(H @ x, x) (apart from the slicing)
        H[..., :, n:] -= torch.matmul(H[..., :, n:], xcol) * xrow

    D[..., -1] = (-1) ** (dim - 1) * D[..., :-1].prod(axis=-1)

    # Without vectorization this could be written as H = diag(D) @ H,
    # left-multiplication by a diagonal matrix amounts to multiplying each
    # row of H by an element of the diagonal, so we add a dummy axis for
    # the column index
    H *= D[..., :, None]
    return H
