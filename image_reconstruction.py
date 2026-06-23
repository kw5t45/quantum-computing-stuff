import matplotlib.pyplot as plt
import numpy as np


def reconstruct_nass_state(state, nx, ny, original_norm, plot=False):
    """

    :param state: given nass state
    :param nx: number of x qubits
    :param ny: number of y qubits
    :param original_norm:  original normalization
    :param plot: plot or not
    :return:
    """
    W = 2**nx
    H = 2**ny
    img = np.zeros((H, W))

    for idx, amp in enumerate(state):
        prob = np.abs(amp)**2
        if prob == 0:
            continue

        bits = np.binary_repr(idx, width=nx + ny)
        y_bits = bits[:ny]
        x_bits = bits[ny:]
        y = int(y_bits, 2)
        x = int(x_bits, 2)

        img[y, x] = np.sqrt(prob) * original_norm   # <-- FIX

    if plot:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.title("Reconstructed NASS Image")
        plt.colorbar()
        plt.show()

    return img


def reconstruct_neqr_state(state, b, nx, ny, plot=False):

    W = 2**nx
    H = 2**ny

    img = np.zeros((H,W))

    for idx, amp in enumerate(state):

        prob = np.abs(amp)**2

        if prob == 0:
            continue

        bits = np.binary_repr(
            idx,
            width=b+nx+ny
        )

        g_bits = bits[:b]

        y_bits = bits[b:b+ny]

        x_bits = bits[b+ny:]

        gray = int(g_bits,2)

        y = int(y_bits,2)

        x = int(x_bits,2)

        img[y,x] += gray * prob

    # compensate for equal-position superposition
    N = 2**(nx+ny)

    img = img * N

    if plot:

        plt.imshow(
            img,
            cmap="gray",
            vmin=0,
            vmax=(2**b)-1
        )

        plt.show()

    return img

def reconstruct_frqi_state(qml_state, n):
    dim = 2 ** n
    num_pixels = dim * dim
    image = np.zeros((dim, dim))

    for idx in range(num_pixels):
        row = idx // dim
        col = idx % dim

        zero_idx = idx * 2  # color qubit = 0
        one_idx = idx * 2 + 1  # color qubit = 1

        amp0 = qml_state[zero_idx]
        amp1 = qml_state[one_idx]

        # Normalize (account for 1/sqrt(num_pixels) factor)
        norm = np.sqrt(np.abs(amp0) ** 2 + np.abs(amp1) ** 2)
        amp0 /= norm
        amp1 /= norm

        # Extract theta from the amplitudes
        # |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
        # So θ = arctan2(|amp1|, |amp0|) or arcsin(|amp1|)
        theta = np.arctan2(np.abs(amp1), np.abs(amp0))  # θ ∈ [0, π/2]

        # Convert back to grayscale: g = (θ / (π/2)) × 255
        image[row, col] = (theta / (np.pi / 2)) * 255

    return image

