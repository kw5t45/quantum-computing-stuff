from image_qcht import qcht_on_position_registers, inverse_qcht_image_neqr
import numpy as np
from PIL import Image
from state_preparation import neqr_encode_image
from image_reconstruction import reconstruct_neqr_state, reconstruct_frqi_state, reconstruct_nass_state


path = "resources/baboon.png"

img = Image.open(path).convert("L")  # "L" = 8-bit grayscale
img_resized = img.resize((128, 128), Image.BICUBIC)

# Convert to numpy array
img_array_o = np.array(img_resized)
#####################

state_neqr, b, neqr_x, neqr_y = neqr_encode_image(path)

n_qubits_image_total = b + neqr_x + neqr_y  # + 1 auxillary is added in function
transformed_image = qcht_on_position_registers(state_neqr, b, neqr_x, neqr_y)
print(len(transformed_image))
##################
reconstructed_image = inverse_qcht_image_neqr(transformed_image, b, neqr_x, neqr_y)
# Extract only states where auxiliary qubit is |0⟩
n_data_qubits = b + neqr_x + neqr_y
reconstructed_image_traced = np.zeros(2 ** n_data_qubits, dtype=complex)

for i in range(2 ** n_data_qubits):
    # Index where auxiliary (last qubit) is 0
    reconstructed_image_traced[i] = reconstructed_image[i * 2]

# Renormalize
reconstructed_image_traced = reconstructed_image_traced / np.linalg.norm(reconstructed_image_traced)

img_ = reconstruct_neqr_state(reconstructed_image_traced, b, neqr_x, neqr_y, plot=False)
##########################

import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

img_orig = np.array(img_array_o, dtype=np.float32)
img_recon = np.array(img_, dtype=np.float32)

img_recon_normalized = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())
img_recon_scaled = img_recon_normalized * (img_orig.max() - img_orig.min()) + img_orig.min()

# 1️⃣ Compute MSE map
mse_map = (img_orig - img_recon_scaled) ** 2

# 2️⃣ Compute overall metrics
mse_val = mean_squared_error(img_orig, img_recon_scaled)
psnr_val = peak_signal_noise_ratio(img_orig, img_recon_scaled, data_range=255)
ssim_val = structural_similarity(img_orig, img_recon_scaled, data_range=255)

print(f"MSE: {mse_val:.4f}")
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")

# 3️⃣ Plot MSE map + original and reconstructed
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(img_orig, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(img_recon_scaled, cmap='gray')
axes[1].set_title("Reconstructed Image")
axes[1].axis('off')

im = axes[2].imshow(mse_map, cmap='hot')
axes[2].set_title("MSE Map")
axes[2].axis('off')
fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle(f"MSE={mse_val:.2f}, PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}", fontsize=16)
plt.show()
