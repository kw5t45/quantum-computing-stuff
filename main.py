from PIL import Image
import numpy as np
from timing import timer

from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity
)

from image_qcht import convert_deconvert_image_qcht_nass, convert_deconvert_image_qcht_neqr, \
    convert_deconvert_image_qcht_frqi

path = "resources/test_images/tank2.png"

# ground truth image
gt = Image.open(path).convert("L")
gt = gt.resize((128,128), Image.BICUBIC)
gt = np.array(gt, dtype=np.float64)

# reconstructed images
nass_image = timer(convert_deconvert_image_qcht_nass)(path)
neqr_image = timer(convert_deconvert_image_qcht_neqr)(path)
frqi_image = timer(convert_deconvert_image_qcht_frqi)(path)


def evaluate(name, reconstructed, ground_truth):

    reconstructed = np.array(
        reconstructed,
        dtype=np.float64
    )

    mse = mean_squared_error(
        ground_truth,
        reconstructed
    )

    psnr = peak_signal_noise_ratio(
        ground_truth,
        reconstructed,
        data_range=255
    )

    ssim = structural_similarity(
        ground_truth,
        reconstructed,
        data_range=255
    )

    print(f"\n{name}")
    print(f"MSE  : {mse:.4f}")
    print(f"PSNR : {psnr:.4f} dB")
    print(f"SSIM : {ssim:.4f}")
    print("==Exponential form==")
    print(f"MSE  : {mse:.4e}")
    print(f"PSNR : {psnr:.4e} dB")
    print(f"SSIM : {ssim:.4e}")

evaluate("NASS", nass_image, gt)
evaluate("NEQR", neqr_image, gt)
evaluate("FRQI", frqi_image, gt)
