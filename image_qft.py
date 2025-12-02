from plots import *
import math
from PIL import Image
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

img = Image.open(r"resources/baboon.png").convert('L')
img = img.resize((128, 128), Image.BICUBIC)
img_array = np.array(img)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Input")
plt.show()

rows, cols = img_array.shape
pixels = rows * cols

img_1d = img_array.flatten()
amplitudes = img_1d / np.linalg.norm(img_1d)

# x, y, and total qubits
nx = math.ceil(math.log2(rows))
ny = math.ceil(math.log2(cols))
n_qubits = nx + ny

# state size and norm
state = np.zeros(2 ** n_qubits, dtype=complex)
state[:pixels] = amplitudes
state /= np.linalg.norm(state)

print("Total qubits:", n_qubits)
print("Amplitude vector length:", len(state))
print("Image state:", state)

dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qft_2d(state):
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    # row qubit qft
    qml.QFT(wires=range(nx))

    # col qubits qft
    qml.QFT(wires=range(nx, nx + ny))

    return qml.state()


state_qft = qft_2d(state)
print(state_qft)


def classic_fft(img_array, shift=True, magnitude=True, log_scale=True, reshape=True, reshape_size=(128, 128)):
    """

    :param img_array: array representing an image - pixel value intensities, greyscale for current version. didnt add
    assertions yet so code might explode if colorful image etc.
    :param shift: shift DC comp to centre
    :param magnitude: plot magnitude of fft output
    :param log_scale: log scale the output
    :param reshape: reshape back to square - ready for plotting
    :param reshape_size: reshape size.
    :return:
    """
    # Compute the 2D FFT
    fft = np.fft.fft2(img_array)

    # Shift zero frequency to center
    if shift:
        fft = np.fft.fftshift(fft)

    # Convert to magnitude
    if magnitude:
        fft = np.abs(fft)

    # Log scale improves visibility
    if log_scale:
        fft = np.log1p(fft)

    if reshape:
        return fft.reshape(reshape_size)
    return fft


# image reconstruct
def reconstruct_image_from_state(state, fft_shift=True, magnitude=True, log_scale=True, reshape=True, shape=(128, 128)):
    """

    :param state: NASS state encoding image in amplitudes, after QFT is applied e.g. [0.00499946+0.j 0.0040112 +0.j 0.00441813+0.j...]

    :param fft_shift: shift to centre
    :param magnitude: take magnitude of state
    :param log_scale: log scale for plotting, else dc can be seen only
    :param reshape: reshape to shape param
    :param shape:  reshape param
    :return:
    """
    num = shape[0] * shape[1]
    amps = state[:num]

    # Get magnitudes (not power)
    if magnitude:
        freqs = np.abs(amps)
    else:
        freqs = amps.real  # or however you want to handle complex values

    if reshape:
        freqs = freqs.reshape(shape)

    # FFT shift
    if fft_shift:
        freqs = np.fft.fftshift(freqs)

    # log scaling for visualization
    if log_scale:
        freqs = np.log(freqs + 1e-10)  # ?????????

    return freqs

reconstructed = reconstruct_image_from_state(state_qft)
# plot_probability_distribution(state_qft, n_qubits=n_qubits)

plt.imshow(classic_fft(img_array), cmap='gray')
plt.axis('off')
plt.title("Classic FFT result")
plt.show()

plt.imshow(reconstructed, cmap='gray')
plt.axis('off')
plt.title("QFT result")
plt.show()
