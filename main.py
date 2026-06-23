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
#nass_image = timer(convert_deconvert_image_qcht_nass)(path)
#neqr_image = timer(convert_deconvert_image_qcht_neqr)(path)
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

#evaluate("NASS", nass_image, gt)
#evaluate("NEQR", neqr_image, gt)
frqi_image = frqi_image.reshape(gt.shape)

print(frqi_image.shape, gt.shape)
print(frqi_image)
evaluate("FRQI", frqi_image, gt)


import matplotlib.pyplot as plt


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(gt, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow((frqi_image), cmap="gray")
plt.title("Reconstructed")
plt.axis("off")

plt.tight_layout()
plt.show()


plt.subplot(1, 2, 1)
plt.hist(gt.ravel(), bins=256, range=(0, 256), color="blue", alpha=0.7)
plt.title("Ground Truth Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(np.real(frqi_image).ravel(), bins=256, range=(0, 256), color="orange", alpha=0.7)
plt.title("Reconstructed Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()