import numpy as np
import matplotlib.pyplot as plt


def plot_phase_output(state, n_qubits, save=False, save_name='phase_img'):
    phases = np.angle(state)

    plt.bar(range(2 ** n_qubits), phases)
    plt.xlabel("Basis state |k>")
    plt.ylabel("Phase (radians)")
    plt.title("Phase distribution after QFT")

    if save:
        plt.savefig(f'{save_name}.png')
    plt.show()


def plot_probability_distribution(state, n_qubits, save=False, save_name='prob_img'):
    probabilities = np.abs(state) ** 2

    plt.bar(range(2 ** n_qubits), probabilities)
    plt.xlabel("Basis state |k>")
    plt.ylabel("Probability")
    plt.title("Probability distribution after QFT")

    if save:
        plt.savefig(f'{save_name}.png')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def plot_four_images_with_mse_maps(img1, img2, img3, img4, img5):
    """
    2-row figure:
      Row 1: images (Original, NEQR, NASS, FFT, FRQI)
      Row 2: MSE maps, except column 1 which shows PSNR + SSIM table
    """

    H, W = img2.shape
    N = H * W

    # --- Properly scale NEQR ---
    img2_scaled = img2 * N

    # --- Ensure FRQI is scaled to [0,255] ---
    img5_scaled = img5

    # --- Compute MSE maps ---
    diff12 = (img1 - img2_scaled) ** 2
    diff13 = (img1 - img3) ** 2
    diff14 = (img1 - img4) ** 2
    diff15 = (img1 - img5_scaled) ** 2

    mse12 = np.mean(diff12)
    mse13 = np.mean(diff13)
    mse14 = np.mean(diff14)
    mse15 = np.mean(diff15)

    # --- Compute PSNR & SSIM ---
    psnr12 = psnr(img1, img2_scaled, data_range=255)
    psnr13 = psnr(img1, img3, data_range=255)
    psnr14 = psnr(img1, img4, data_range=255)
    psnr15 = psnr(img1, img5_scaled, data_range=255)

    ssim12 = ssim(img1, img2_scaled, data_range=255)
    ssim13 = ssim(img1, img3, data_range=255)
    ssim14 = ssim(img1, img4, data_range=255)
    ssim15 = ssim(img1, img5_scaled, data_range=255)

    # --- Metrics table ---
    metrics_text = (
        "   Method    |   PSNR   |   SSIM   \n"
        "-----------------------------------\n"
        f" NEQR        | {psnr12:8.2f} | {ssim12:7.4f}\n"
        f" NASS        | {psnr13:8.2f} | {ssim13:7.4f}\n"
        f" FFT         | {psnr14:8.2f} | {ssim14:7.4f}\n"
        f" FRQI        | {psnr15:8.2f} | {ssim15:7.4f}"
    )

    plt.figure(figsize=(26, 10))

    # ----------------------------
    # Row 1: Images
    # ----------------------------
    titles_row1 = [
        "Original image",
        "NEQR reconstructed image (scaled)",
        "NASS reconstructed image",
        "FFT reconstructed image",
        "FRQI reconstructed image"
    ]

    images_row1 = [
        img1,
        img2_scaled,
        img3,
        img4,
        img5_scaled
    ]

    for i, (title, img) in enumerate(zip(titles_row1, images_row1), start=1):
        plt.subplot(2, 5, i)
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")

    # ----------------------------
    # Row 2: Column 1 → Metrics table
    # ----------------------------
    ax = plt.subplot(2, 5, 6)
    ax.axis("off")
    ax.text(
        0.0, 0.5,
        metrics_text,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
    )
    ax.set_title("Quality Metrics\nvs Original")

    # ----------------------------
    # Row 2: MSE Maps
    # ----------------------------
    mse_maps = [diff12, diff13, diff14, diff15]
    mse_vals = [mse12, mse13, mse14, mse15]
    mse_titles = [ # scientific notation assuming small mses which is the case
        f"MSE Map: NEQR vs Original\nMSE = {mse12:.2e}",
        f"MSE Map: NASS vs Original\nMSE = {mse13:.2e}",
        f"MSE Map: FFT vs Original\nMSE = {mse14:.2e}",
        f"MSE Map: FRQI vs Original\nMSE = {mse15:.2e}",
    ]
    for i in range(4):
        plt.subplot(2, 5, 7 + i)
        plt.imshow(mse_maps[i], cmap="hot")
        plt.title(mse_titles[i])
        plt.colorbar()
        plt.axis("off")

    plt.tight_layout()
    plt.show()