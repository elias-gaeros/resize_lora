import torch

from loralib.merge import ties, tucker_spectrum


def test_ties_trims_elects_signs_and_averages_agreeing_entries():
    cores = torch.tensor(
        [
            [[3.0, -1.0], [0.5, 2.0]],
            [[-2.0, -4.0], [1.0, 0.1]],
        ]
    )

    # floor(4 * 0.5) + 1 = 3 kept, as in the original CtM code: [3, -1, 0, 2] and
    # [-2, -4, 1, 0]. Elected signs: +, -, +, +; -1 and -4 agree and are averaged.
    merged = ties(cores, density=0.5)

    torch.testing.assert_close(merged, torch.tensor([[3.0, -2.5], [1.0, 2.0]]))


def test_tucker_spectrum_of_one_matrix_is_its_singular_values():
    torch.manual_seed(0)
    matrix = torch.randn(1, 6, 5)

    _, _, s, _, _ = tucker_spectrum(matrix)

    torch.testing.assert_close(s, torch.linalg.svdvals(matrix[0]))


def test_tucker_spectrum_holds_the_energy_of_all_matrices():
    torch.manual_seed(0)
    cores = torch.randn(3, 8, 5)

    _, _, s, _, _ = tucker_spectrum(cores)

    # So that the recipes' errors, norm(discarded s) / norm(s), are relative to all LoRAs
    torch.testing.assert_close(s.square().sum(), cores.square().sum())
