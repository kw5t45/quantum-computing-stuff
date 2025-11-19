from plots import *
import math
from PIL import Image
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

# mnist 5 image
img = Image.open(r"resources/mnist_digit_5.png").convert('L')
img_array = np.array(img)

rows, cols = img_array.shape  # 28x28
pixels = rows * cols

img_1d = img_array.flatten()
amplitudes = img_1d / np.linalg.norm(img_1d)


plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Input")
plt.show()

# x, y, and total qubits
nx = math.ceil(math.log2(rows))  # 5
ny = math.ceil(math.log2(cols))  # 5
n_qubits = nx + ny               # 10

# state size and norm
state = np.zeros(2 ** n_qubits, dtype=complex)
state[:pixels] = amplitudes
state /= np.linalg.norm(state)

print("Total qubits:", n_qubits)
print("Amplitude vector length:", len(state))


dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qft_2d(state):
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    # QFT on row qubits (first nx)
    qml.QFT(wires=range(nx))

    # QFT on column qubits (last ny)
    qml.QFT(wires=range(nx, nx+ny))

    return qml.state()


state_qft = qft_2d(state)
print(state_qft)

# image reconstruct
def reconstruct_image_from_state(state, shape=(28, 28)):
    num = shape[0] * shape[1]
    amps = state[:num]

    # magnitudes = frequency intensity
    img = np.abs(amps)
    img /= np.max(img)

    return img.reshape(shape)

reconstructed = reconstruct_image_from_state(state_qft)

plot_probability_distribution(state_qft, n_qubits, save=True, save_name='2d_qft_mnist')