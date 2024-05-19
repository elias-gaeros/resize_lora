import torch


def fast_decompose(up, down):
    Ud, Sd, Vhd = torch.linalg.svd(down.flatten(start_dim=1), full_matrices=False)
    Uu, Su, Vhu = torch.linalg.svd(up.flatten(start_dim=1), full_matrices=False)
    Uc, Sc, Vhc = torch.linalg.svd((Vhu * Su.unsqueeze(1)) @ (Ud * Sd))
    U = Uu @ Uc
    Vh = Vhc @ Vhd
    return U, Sc, Vh


def load_lora_layer(lora_file, name, **to_kwargs):
    alpha = lora_file.get_tensor(f"{name}.alpha").item()
    down = lora_file.get_tensor(f"{name}.lora_down.weight").to(**to_kwargs)
    up = lora_file.get_tensor(f"{name}.lora_up.weight").to(**to_kwargs)
    return alpha, up, down


def outer_cosine_sim(U1, U2):
    U1n = U1 / torch.linalg.norm(U1, dim=0)
    U2n = U2 / torch.linalg.norm(U2, dim=0)
    return U1n.T @ U2n
