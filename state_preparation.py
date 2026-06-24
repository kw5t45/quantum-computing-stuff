from plots import *
import math
from PIL import Image
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

def nass_encode_image(path, size=(128, 128)):
    """

    :param path: path of img
    :param size: size of img
    :return: nass encoded complex array representing given image.
    """

    img = Image.open(rf"{path}").convert('L')
    img = img.resize(size, Image.BICUBIC)
    img_array = np.array(img)

    rows, cols = img_array.shape
    pixels = rows * cols

    img_1d = img_array.flatten()
    original_norm = np.linalg.norm(img_1d)
    amplitudes = img_1d / original_norm

    # x, y, and total qubits
    nx = math.ceil(math.log2(rows))
    ny = math.ceil(math.log2(cols))
    n_qubits = nx + ny

    # state size and norm
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[:pixels] = amplitudes
    state /= np.linalg.norm(state)


    return state, nx, ny, original_norm

def neqr_encode_image(path, size=(128, 128)) -> tuple:
    """

    :param path: load img from path
    :param size: size of image
    :return: np array state vector representing neqr image. should be size 2^(b + nx + ny)
    :return: number of color qubits - 8
    :return: number of x --
    :return: number of y --

    """
    """
    Load a grayscale image and return its NEQR statevector.

    Returns:
        statevector: complex numpy array of size 2^(b + nx + ny)
        b : number of pixel-value qubits (8)
        nx: number of x-position qubits
        ny: number of y-position qubits
    """

    # ----------------
    # Load image
    # ----------------
    img = Image.open(path).convert('L')
    img = img.resize(size, Image.BICUBIC)
    img_array = np.array(img)

    rows, cols = img_array.shape

    # Pixel-value qubits (8 bits for grayscale 0–255)
    b = 8

    # Position qubits
    nx = math.ceil(math.log2(rows))
    ny = math.ceil(math.log2(cols))

    # Total qubits
    N = b + nx + ny

    dim = 2 ** N

    state = np.zeros(dim, dtype=np.complex128)


    # Bit layout: |f0 f1 ... f7  x_{nx-1} ... x_0  y_{ny-1} ... y_0>

    for x in range(rows):
        for y in range(cols):
            pixel_val = img_array[x, y]  # 0..255

            x_bits = x << ny
            xy_bits = x_bits | y

            # Combine to full basis index
            index = (int(pixel_val) << (nx + ny)) | xy_bits
            # Amplitude = 1 / sqrt(rows*cols)
            state[index] = 1.0

    # Normalize
    state /= np.linalg.norm(state)

    return state, b, nx, ny

def frqi_encode_image(path, size=(128, 128)) -> tuple:
    """

    :param path: path of image
    :param size: size of image which should be 2^n
    :return: state: complex frqi state vector
    :return:  n: number of qubits for each axis
    SHOULD BE 15 TOTAL QUBITS FOR 128X128 IMAGE, = 32768 AMPLITUDES
    """

    # Load image in grayscale
    img = Image.open(path).convert('L')
    img = img.resize(size, Image.BICUBIC)
    img_array = np.array(img, dtype=float)

    # dimensions must be 2^n x 2^n
    rows, cols = img_array.shape
    if rows != cols or (rows & (rows - 1)) != 0:
        raise ValueError("FRQI requires dimensions 2^n × 2^n")

    n = int(math.log2(rows))
    n_pixels = rows * cols

    # Flatten pixels
    pixels = img_array.flatten()

    # Convert pixel intensities [0..255] → angles θ_i in [0..π/2]
    thetas = (np.pi / 2) * (pixels / 255.0)

    # FRQI state vector size is 2^(2n+1)
    total_states = 2 ** (2 * n + 1)

    # Output FRQI state vector
    state = np.zeros(total_states, dtype=complex)

    # Populate amplitude for each pixel position i:
    # |i,0> = cos(theta_i)
    # |i,1> = sin(theta_i)
    for i in range(n_pixels):
        state[2 * i]     = np.cos(thetas[i])  # |0⟩ color qubit
        state[2 * i + 1] = np.sin(thetas[i])  # |1⟩ color qubit

    # Normalize entire state vector
    state = state / np.linalg.norm(state)
    return state, n