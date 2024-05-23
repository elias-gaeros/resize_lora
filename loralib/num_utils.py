import torch


def fast_decompose(up, down):
    Qu, Ru = torch.linalg.qr(up.flatten(start_dim=1))
    Qd, Rd = torch.linalg.qr(down.flatten(start_dim=1).mT)
    Uc, Sc, Vhc = torch.linalg.svd(Ru @ Rd.mT)
    return Qu @ Uc, Sc, Vhc @ Qd.mT


def load_lora_layer(lora_file, name, **to_kwargs):
    alpha = lora_file.get_tensor(f"{name}.alpha").item()
    down = lora_file.get_tensor(f"{name}.lora_down.weight").to(**to_kwargs)
    up = lora_file.get_tensor(f"{name}.lora_up.weight").to(**to_kwargs)
    return alpha, up, down


def outer_cosine_sim(U1, U2):
    U1n = U1 / torch.linalg.norm(U1, dim=0)
    U2n = U2 / torch.linalg.norm(U2, dim=0)
    return U1n.T @ U2n
