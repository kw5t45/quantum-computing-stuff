from PIL import Image
import pennylane as qml
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


n_qubits = 22
dev = qml.device("default.qubit", wires=22)


@qml.qnode(dev)
def qft_2d_nass(state, nx, ny):
    """

    :param state: nass state vector of image
    :param nx: number of  x qubits
    :param ny:  --- y bits
    :return: applies qft on 2d nsas state
    """
    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    # row qubit qft
    qml.QFT(wires=range(nx))

    # col qubits qft
    qml.QFT(wires=range(nx, nx + ny))

    return qml.state()

def qft_2d_frqi(state, n):
    """

    :param state: frqi state vector len 2^(2n+1)
    :param n: number of x and y qubits (shouldl be log2 image dim - 7 for 128x128)
    :return: nd array of qft transformed state.
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

dev = qml.device("default.qubit", wires=22)
@qml.qnode(dev)
def qft_2d_neqr(state, b, nx, ny):
    """

    :param state: neqr state vector
    :param b: number of color qubits
    :param nx: number of x qubits
    :param ny:  --- y qubits
    :return: applies qft on state
    """
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

    :param fft_state: fourier domain data produced by classic fft
    :param shifted: whether fft was shifted to centre
    :param magnitude: if true, fft_state contains only magnitude, which then requires original phase to reconstruct
    :param log_scaled: if was log1p
    :param original_phase: if mag=True
    :param reshape: bool to reshape to size (int int)
    :param reshape_size:  --
    :return: real valued reconstructed img aarray
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
    Returns the frequency domain of the image.
    :param state: qft transformed state vector - should be length 2^(b +nx + ny)
    :param b: color value cubits
    :param nx:
    :param ny:
    :return: complex array (img)
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

    :param frqi_state: frqi state after qft is applied.
    :param n: number of qubits
    :return: original frqi state which (should) represent encoded image
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

@qml.qnode(dev)
def inverse_qft_2d_neqr(qft_state, b, nx, ny):
    """

    :param qft_state: qft state
    :param b: greyscale bits
    :param nx: x bits
    :param ny: y bits
    :return: neqr state after application of 2d qft
    """

    n_qubits = b + nx + ny

    # Load QFT-transformed state
    qml.StatePrep(qft_state, wires=range(n_qubits), normalize=True)

    # Inverse QFT on x-qubits (columns)
    qml.adjoint(qml.QFT)(wires=range(b + ny, b + ny + nx))

    # Inverse QFT on y-qubits (rows)
    qml.adjoint(qml.QFT)(wires=range(b, b + ny))


    return qml.state()



#
#
#
# # FRQI image state
#
# state, n = frqi_encode_image(path, size=(128,128))
# qft_state = qft_2d_frqi(state, n)
# inv_state = inverse_qft_frqi(qft_state, n)
# frqi_reconstructed = frqi_state_to_image(inv_state, n)
#
# # NEQR image
# state_neqr, b, neqr_x, neqr_y = neqr_encode_image(path)
# print(f'encoded state len {len(state_neqr)}')
# qft_neqr_state = qft_2d_neqr(state_neqr, b, neqr_x, neqr_y)
# print(f'encoded state after qft len {len(state_neqr)}')
# neqr_reconstructed = inverse_qft_2d_neqr(qft_neqr_state, b, neqr_x, neqr_y)
# print(f'encoded state after inv qft len {len(state_neqr)}')
# neqr_corr = plot_neqr_state(neqr_reconstructed, b, neqr_x, neqr_y, plot=False)
# print(f'encoded state after img reconstruction len {len(state_neqr)}')
# # NASS state
# state_nass, nass_x, nass_y = nass_encode_image(path)
# original_norm = np.linalg.norm(img_array_o.flatten())
# nass_corr = plot_nass_state(state_nass, nass_x, nass_y, plot=False, original_norm=original_norm)
#
#
# # classic FFT
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

# plot_four_images_with_mse_maps(
#     img_array_o,
#     neqr_corr,
#     nass_corr,
#     img_rec_fft,
#     frqi_reconstructed
# )