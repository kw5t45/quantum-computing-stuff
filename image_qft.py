from plots import *
import math
from PIL import Image
import numpy as np
import pennylane as qml

img = Image.open("mnist_digit_5.png").convert('L')  # 'L' converts to grayscale

img_array = np.array(img)

img_1d = img_array.flatten()

amplitudes = img_1d / np.linalg.norm(img_1d)

n_qubits = math.ceil(math.log(len(amplitudes), 2))

state = np.zeros(2 ** n_qubits, dtype=complex)
state[:len(amplitudes)] = amplitudes
state = state / np.linalg.norm(state)

print(len(state))
print(state)

# plot_probability_distribution(state, 10)

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qft_superposition(n_qubits, state):
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    qml.QFT(wires=range(n_qubits))

    return qml.state()


# plot_probability_distribution(state, n_qubits, save=True, save_name='img_superposition_state')

state_qft = qft_superposition(n_qubits, state=state)

# plot_probability_distribution(state_qft, n_qubits, save=True, save_name='qft_superosition_state')
# plot_phase_output(state_qft, n_qubits, save=True, save_name='qft_phase')
#

def reconstruct_image_from_state(state, image_shape=(28, 28)):
    """
    reconstruct a grayscale image from a quantum state vector.

    state (np.ndarray): complex quantum state vector (length >= product of image_shape)
    image_shape (tuple): shape of the original image

    return np.ndarray: Reconstructed 2D grayscale image with values 0-1.
    """
    # take the first pixels (if padded)
    num_pixels = image_shape[0] * image_shape[1]
    amplitudes = state[:num_pixels]

    # convert amplitudes to pixel intensities
    pixels = np.abs(amplitudes)

    # normalize to 0-1
    pixels = pixels / np.max(pixels)

    # Reshape to 2D image
    img = pixels.reshape(image_shape)

    return img

reconstructed_img = reconstruct_image_from_state(state_qft)

plt.imshow(reconstructed_img, cmap='gray')
plt.axis('off')
plt.show()