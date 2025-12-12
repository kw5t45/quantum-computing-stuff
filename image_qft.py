from plots import *
import math
from PIL import Image
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

path = "resources/baboon.png"

img = Image.open(path).convert("L")   # "L" = 8-bit grayscale
img_resized = img.resize((128, 128), Image.BICUBIC)

# Convert to numpy array
img_array_o = np.array(img_resized)

# Plot
# plt.imshow(img_array_o, cmap="gray", vmin=0, vmax=255)
# plt.title("Grayscale Image")
# plt.axis("off")
# plt.show()

def nass_encode_image(path, size=(128, 128)):

    img = Image.open(rf"{path}").convert('L')
    img = img.resize(size, Image.BICUBIC)
    img_array = np.array(img)

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


    return state, nx, ny

def neqr_encode_image(path, size=(128, 128)):
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

    # Dimension of NEQR Hilbert space
    dim = 2 ** N

    # Initialize statevector
    state = np.zeros(dim, dtype=np.complex128)

    # ----------------
    # Encode NEQR basis states
    # ----------------
    # Bit layout: |f0 f1 ... f7  x_{nx-1} ... x_0  y_{ny-1} ... y_0>
    #
    # For each pixel (x,y), we build its integer basis index.
    # ----------------

    for x in range(rows):
        for y in range(cols):
            pixel_val = img_array[x, y]  # 0..255

            # Bits
            f_bits = pixel_val
            x_bits = x << ny
            xy_bits = x_bits | y

            # Combine to full basis index
            index = (f_bits << (nx + ny)) | xy_bits

            # Amplitude = 1 / sqrt(rows*cols)
            state[index] = 1.0

    # Normalize
    state /= np.linalg.norm(state)

    return state, b, nx, ny

def frqi_encode_image(path, size=(128, 128)):
    """
    Converts an image into an FRQI state vector representation.

    Parameters:
        path (str): Image file path.
        size (tuple): Image resize dimensions (must be 2^n × 2^n).

    Returns:
        state (np.ndarray): Complex FRQI quantum state vector.
        n (int): Number of qubits for each axis (image is 2^n × 2^n).
    """

    # Load image in grayscale
    img = Image.open(path).convert('L')
    img = img.resize(size, Image.BICUBIC)
    img_array = np.array(img, dtype=float)

    # Dimensions must be 2^n x 2^n
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

n_qubits = 22
dev = qml.device("default.qubit", wires=22)


@qml.qnode(dev)
def qft_2d_nass(state, nx, ny):
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    # row qubit qft
    qml.QFT(wires=range(nx))

    # col qubits qft
    qml.QFT(wires=range(nx, nx + ny))

    return qml.state()

def qft_2d_frqi(state, n):
    """
    Applies a 2D QFT to an FRQI-encoded quantum image.

    Parameters:
        state (np.ndarray): FRQI state vector of length 2^(2n+1)
        n (int): log2(image dimension), e.g. n=7 for 128x128

    Returns:
        np.ndarray: The QFT-transformed state vector.
    """

    n_qubits = 1 + 2 * n                       # 1 color + 2n position qubits
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        # Load the FRQI state
        qml.StatePrep(state, wires=range(n_qubits), normalize=True)

        # Position qubits only (skip the color qubit)
        row_wires = list(range(1, 1 + n))
        col_wires = list(range(1 + n, 1 + 2*n))

        # 2D QFT: QFT on rows, then QFT on columns
        qml.QFT(wires=row_wires)
        qml.QFT(wires=col_wires)

        return qml.state()

    return circuit()

dev = qml.device("default.qubit", wires=15)
@qml.qnode(dev)
def qft_2d_neqr(state, b, nx, ny):
    n_qubits = b + nx + ny

    # Load NEQR/NASS state
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    # Apply QFT on y-qubits (row dimension)
    qml.QFT(wires=range(b, b + ny))

    # Apply QFT on x-qubits (column dimension)
    qml.QFT(wires=range(b + ny, b + ny + nx))

    return qml.state()
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

def classic_ifft(fft_state,
                 shifted=True,
                 magnitude=False,
                 log_scaled=False,
                 original_phase=None,
                 reshape=True,
                 reshape_size=(128, 128)):
    """
    Reconstruct an image from a 2D FFT output.

    Parameters
    ----------
    fft_state : ndarray
        The Fourier-domain data produced by classic_fft.
    shifted : bool
        Whether fft_state was fftshifted.
    magnitude : bool
        If True, fft_state contains only magnitude (|F|).
        Requires original_phase to reconstruct.
    log_scaled : bool
        If True, fft_state was log1p() scaled, so undo exp()-1.
    original_phase : ndarray or None
        Required if magnitude=True. Must contain the phase of the original FFT.
    reshape : bool
        Whether to reshape the output to reshape_size.
    reshape_size : (int, int)
        Final output shape.

    Returns
    -------
    img_reconstructed : ndarray
        Real-valued reconstructed image.
    """

    F = fft_state.copy().astype(complex)

    # --- Undo log scaling ---
    if log_scaled:
        F = np.expm1(F)   # inverse of log1p

    # --- Undo magnitude-only (requires phase) ---
    if magnitude:
        if original_phase is None:
            raise ValueError("Reconstruction from magnitude-only FFT requires original_phase.")
        F = F * np.exp(1j * original_phase)

    # --- Undo shift ---
    if shifted:
        F = np.fft.ifftshift(F)

    # --- Inverse FFT ---
    img_complex = np.fft.ifft2(F)

    # Take real part (imag is numerical noise)
    img = np.real(img_complex)

    if reshape:
        img = img.reshape(reshape_size)

    return img

# image reconstruct
def nass_extract_frequency_image(state, fft_shift=True, magnitude=True, log_scale=True, reshape=True, shape=(128, 128)):
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

def neqr_extract_frequency_image(state, b=8, nx=7, ny=7):
    """
    Extract the 128x128 2D Fourier image from the NEQR+QFT statevector.

    state: full QFT-transformed statevector (length 2^(b+nx+ny))
    b: number of value qubits
    nx, ny: number of x and y qubits

    returns: complex array (128 x 128)
    """
    rows = 2**nx
    cols = 2**ny

    freq_img = np.zeros((rows, cols), dtype=np.complex128)

    for kx in range(rows):
        for ky in range(cols):
            base_index = (kx << ny) | ky  # pack frequency bits

            # sum over all 256 possible color states
            val = 0.0j
            for c in range(256):
                idx = (c << (nx + ny)) | base_index
                val += state[idx]

            freq_img[kx, ky] = val

    return freq_img

@qml.qnode(qml.device("default.qubit", wires=15))
def inverse_qft_frqi(frqi_state, n):
    """
    Inverse 2D QFT on FRQI image.
    n: log2(image dimension), e.g. 7 for 128x128
    """
    n_qubits = 1 + 2 * n  # 1 color + 2n position qubits
    qml.StatePrep(frqi_state, wires=range(n_qubits), normalize=True)

    # Position qubits only
    row_wires = list(range(1, 1 + n))
    col_wires = list(range(1 + n, 1 + 2*n))

    # Inverse 2D QFT: rows then columns
    qml.adjoint(qml.QFT)(wires=row_wires)
    qml.adjoint(qml.QFT)(wires=col_wires)

    return qml.state()


def frqi_state_to_image(qml_state, n):
    """
    Convert a quantum state (FRQI) to a 2D image array.

    Args:
        qml_state (array): Quantum state vector after inverse QFT.
        n (int): log2(image dimension), e.g. 7 for 128x128 image.

    Returns:
        image (2D np.array): Image with values in [0,1].
    """
    dim = 2 ** n  # image dimension
    num_pixels = dim * dim
    image = np.zeros((dim, dim))

    # Each amplitude: |color>|position> => color qubit is the first one
    for idx in range(num_pixels):
        # Map idx to row/col
        row = idx // dim
        col = idx % dim

        # Get amplitudes for color qubit being |0> and |1>
        # position index starts from qubit 1
        zero_idx = idx * 2  # color qubit = 0
        one_idx = idx * 2 + 1  # color qubit = 1

        # FRQI encodes color as theta in first qubit: |0> cos(theta) + |1> sin(theta)
        theta = np.arctan2(np.abs(qml_state[one_idx]), np.abs(qml_state[zero_idx]))

        # Map theta to grayscale [0,1]
        image[row, col] = np.sin(theta) ** 2  # sin^2(theta) gives normalized pixel value

    return image

def plot_neqr_state(state, b, nx, ny, plot=False) -> list | np.ndarray:
    """

    :param state: neqr qft DECODED state to be plotted
    :param b: number of grayscale bits
    :param nx:  -- x bits
    :param ny:    y bits
    :return: returns original image array reconstructed from neqr state
    """

    # Image dimensions
    W = 2**nx
    H = 2**ny
    img = np.zeros((H, W))

    # Iterate through all basis states
    for idx, amp in enumerate(state):
        prob = np.abs(amp)**2
        if prob == 0:
            continue

        # Convert basis index → bitstring
        bits = np.binary_repr(idx, width=b + nx + ny)

        # Partition bits: [grayscale][y][x]
        g_bits = bits[:b]
        y_bits = bits[b:b+ny]
        x_bits = bits[b+ny:]

        # Decode values
        gray = int(g_bits, 2)
        y = int(y_bits, 2)
        x = int(x_bits, 2)

        # Accumulate (for superposed states)
        img[y, x] += gray * prob

    N = 2 ** nx * 2 ** ny
    img_scaled = img * N
    if plot:
        plt.imshow(img_scaled, cmap="gray", vmin=0, vmax=2 ** b - 1)
        plt.title("Reconstructed NEQR Image")
        plt.colorbar(label="Pixel Value")
        plt.show()

    return img
def plot_nass_state(state, nx, ny, original_norm, plot=True):
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


state, n = frqi_encode_image(path, size=(128,128))
print(len(state))
qft_state = qft_2d_frqi(state, n)
print(len(qft_state))
inv_state = inverse_qft_frqi(qft_state, n)

# Convert the quantum state to a 2D image
reconstructed_img = frqi_state_to_image(inv_state, n)

# Plot the reconstructed image
plt.figure(figsize=(6,6))
plt.imshow(reconstructed_img, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title("Reconstructed FRQI Image")
plt.show()
# state_neqr, b, neqr_x, neqr_y = neqr_encode_image(path)
# state_nass, nass_x, nass_y = nass_encode_image(path)
# psi_qft = qft_2d_neqr(state_neqr, b, neqr_x, neqr_y)
#
# psi_original = inverse_qft_2d_neqr(psi_qft, b, neqr_x, neqr_y)
#
#
#
# neqr_corr = plot_neqr_state(psi_original, b, neqr_x, neqr_y, plot=False)
#
# original_norm = np.linalg.norm(img_array_o.flatten())
#
# nass_corr = plot_nass_state(state_nass, nass_x, nass_y, plot=False, original_norm=original_norm)
#
# fft_img = classic_fft(img_array_o,
#                       shift=True,
#                       magnitude=False,
#                       log_scale=False)
# phase = np.angle(np.fft.fft2(img_array_o))
#
# img_rec_fft = classic_ifft(fft_img,
#                        shifted=True,
#                        magnitude=False,
#                        log_scaled=False)
#
#
# plot_four_images_with_mse_maps(img_array_o, neqr_corr, nass_corr, img_rec_fft)
