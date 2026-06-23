import pennylane as qml
import numpy as np
from PIL import Image

from image_reconstruction import reconstruct_neqr_state, reconstruct_nass_state
from state_preparation import neqr_encode_image, nass_encode_image, frqi_encode_image
from timing import timer


def CNOT_ladder(N):
    for wire in range(1, N + 1):
        qml.CNOT([0, wire])


pi = np.pi


def rotate_phases(N):
    """Rotates and shifts the phase of the auxiliary qubit and rotates the jth qubit by
    pi/2^(j+1) in Z."""
    qml.RZ(-pi * (2 ** N - 1) / 2 ** (N + 1), wires=0)
    qml.PhaseShift(-pi / 2 ** (N + 1), wires=0)
    for wire in range(1, N + 1):
        qml.RZ(pi / 2 ** (wire + 1), wires=wire)


def permute_elements(N):
    """Reorders amplitudes of the conditioned states."""
    for wire in reversed(range(1, N + 1)):
        control_wires = [0] + list(range(wire + 1, N + 1))
        qml.MultiControlledX(wires=(*control_wires, wire))


def adjust_phases(N):
    """Adjusts the phase of the auxiliary qubit."""
    qml.RY(-pi / 2, wires=0)
    qml.PhaseShift(-pi / 2, wires=0)

    # First Pauli X gates.
    for wire in range(1, N + 1):
        qml.PauliX(wires=wire)

    # Controlled RX gate.
    qml.ctrl(qml.RX(pi / 2, wires=0), range(1, N + 1))

    # Second Pauli X gates.
    for wire in range(1, N + 1):
        qml.PauliX(wires=wire)


def QChT(wires):
    """Performs the quantum Chebyshev transform on given wires."""

    # If wires is an int, convert to list [0, 1, ..., N]
    if isinstance(wires, int):
        wires = list(range(wires + 1))  # include auxiliary qubit at 0

    N = len(wires) - 1  # auxiliary qubit is first in wires

    aux_wire = wires[0]
    data_wires = wires[1:]

    qml.Hadamard(wires=aux_wire)

    # CNOT ladder
    for wire in data_wires:
        qml.CNOT(wires=[aux_wire, wire])

    qml.QFT(wires=wires)

    # Rotate phases
    qml.RZ(-pi * (2 ** N - 1) / 2 ** (N + 1), wires=aux_wire)
    qml.PhaseShift(-pi / 2 ** (N + 1), wires=aux_wire)
    for i, wire in enumerate(data_wires, start=1):
        qml.RZ(pi / 2 ** (i + 1), wires=wire)

    # Permute elements
    for wire in reversed(data_wires):
        control_wires = [aux_wire] + [w for w in data_wires if w > wire]
        qml.MultiControlledX(wires=(*control_wires, wire))

    # CNOT ladder again
    for wire in data_wires:
        qml.CNOT(wires=[aux_wire, wire])

    # Adjust phases
    qml.RY(-pi / 2, wires=aux_wire)
    qml.PhaseShift(-pi / 2, wires=aux_wire)
    for wire in data_wires:
        qml.PauliX(wires=wire)
    qml.ctrl(qml.RX(pi / 2, wires=aux_wire), data_wires)
    for wire in data_wires:
        qml.PauliX(wires=wire)


dev = qml.device("default.qubit")


@qml.qnode(dev)
def qcht_on_basis_state(N, state=0):
    qml.BasisState(state=state, wires=range(1, N + 1))
    QChT()
    return qml.state()


def circuit_to_draw(N):
    qml.BasisState(state=0, wires=range(1, N + 1))
    QChT()


@qml.qnode(dev)
def qcht_on_state(state, n_qubits):
    qml.StatePrep(state, wires=range(n_qubits + 1), pad_with=0.0)

    # Apply QChT
    QChT(n_qubits)

    return qml.state()


@qml.qnode(dev)
def qcht_on_neqr_image(state, b, nx, ny):
    """
    Applies QChT only on x and y position registers of NEQR image.

    Args:
        state: NEQR statevector (2^(b+nx+ny))
        b: number of color qubits
        nx: number of x qubits
        ny: number of y qubits

    Returns:
        Transformed statevector
    """
    total_qubits = b + nx + ny


    x_wires = list(range(b, b + nx))
    y_wires = list(range(b + nx, total_qubits))
    aux_wire = total_qubits  # auxiliary qubit

    # Prepare full state + auxiliary
    qml.StatePrep(state, wires=range(total_qubits), pad_with=0.0)

    # Apply QChT on x register
    QChT([aux_wire] + x_wires)

    # Apply QChT on y register
    QChT([aux_wire] + y_wires)

    return qml.state()

@qml.qnode(dev)
def qcht_on_nass_image(state, nx, ny):
    """
    Apply QChT on x and y position registers of a NASS image.

    Args:
        state: NASS statevector (size 2^(nx+ny))
        nx: number of x-coordinate qubits
        ny: number of y-coordinate qubits

    Returns:
        transformed statevector
    """

    total_qubits = nx + ny

    x_wires = list(range(nx))
    y_wires = list(range(nx, total_qubits))

    aux_wire = total_qubits

    # Prepare state on position register
    qml.StatePrep(
        state,
        wires=range(total_qubits),
        pad_with=0.0
    )

    # QChT on x register
    QChT([aux_wire] + x_wires)

    # QChT on y register
    QChT([aux_wire] + y_wires)

    return qml.state()

@qml.qnode(dev)
def inverse_from_statevector(statevec, N):
    # Prepare the full state on N+1 qubits using StatePrep
    qml.StatePrep(statevec, wires=range(N + 1))

    # Apply inverse QChT
    qml.adjoint(lambda: QChT(N))()

    return qml.state()


@qml.qnode(dev)
def inverse_qcht_image_neqr(state, b, nx, ny):
    """
    Applies the inverse QChT only on x and y position registers of a NEQR image.

    Args:
        state: NEQR statevector (2^(b+nx+ny))
        b: number of color qubits
        nx: number of x qubits
        ny: number of y qubits

    Returns:
        Transformed statevector (original computational basis)
    """
    total_qubits = b + nx + ny
    # Wires assignment
    color_wires = list(range(b))
    x_wires = list(range(b, b + nx))
    y_wires = list(range(b + nx, total_qubits))
    aux_wire = total_qubits  # auxiliary qubit

    # Prepare full state + auxiliary
    qml.StatePrep(state, wires=range(total_qubits + 1), pad_with=0.0)

    # Apply inverse QChT on x register
    qml.adjoint(lambda: QChT([aux_wire] + y_wires))()

    # Apply inverse QChT on y register
    qml.adjoint(lambda: QChT([aux_wire] + x_wires))()

    return qml.state()
@qml.qnode(dev)
def inverse_qcht_image_nass(state, nx, ny):
    """
    Applies inverse QChT on x and y position registers
    of a NASS encoded image.

    Args:
        state: transformed NASS statevector
        nx: number of x qubits
        ny: number of y qubits

    Returns:
        recovered computational basis statevector
    """

    total_qubits = nx + ny

    # register assignment
    x_wires = list(range(nx))
    y_wires = list(range(nx, total_qubits))

    aux_wire = total_qubits

    # prepare transformed state (+ auxiliary space)
    qml.StatePrep(
        state,
        wires=range(total_qubits + 1),
        pad_with=0.0
    )

    # inverse must occur in reverse order

    # undo y transform
    qml.adjoint(
        lambda: QChT([aux_wire] + y_wires)
    )()

    # undo x transform
    qml.adjoint(
        lambda: QChT([aux_wire] + x_wires)
    )()

    return qml.state()

@qml.qnode(dev)
def qcht_on_frqi_image(state, n):
    """
    Apply QChT on x and y position registers of an FRQI image.

    Args:
        state: FRQI statevector (size 2^(2n+1))
        n: number of qubits per image axis

    Returns:
        transformed statevector
    """

    total_position_qubits = 2 * n
    color_wire = total_position_qubits

    total_qubits = total_position_qubits + 1

    # Position register split:
    # x register -> first n qubits
    # y register -> next n qubits

    x_wires = list(range(n))

    y_wires = list(range(n, 2 * n))

    # auxiliary qubit for QChT
    aux_wire = total_qubits

    # Prepare FRQI state
    qml.StatePrep(
        state,
        wires=range(total_qubits),
        pad_with=0.0
    )

    # Apply QChT to x-position register
    QChT([aux_wire] + x_wires)

    # Apply QChT to y-position register
    QChT([aux_wire] + y_wires)

    return qml.state()

@qml.qnode(dev)
def inverse_qcht_image_frqi(state, n):
    """
    Apply inverse QChT on x and y position registers
    of an FRQI encoded image.

    Args:
        state: transformed FRQI statevector
        n: number of qubits per image axis

    Returns:
        recovered FRQI statevector
    """

    total_position_qubits = 2 * n
    color_wire = total_position_qubits

    total_qubits = total_position_qubits + 1

    # register assignment
    x_wires = list(range(n))

    y_wires = list(range(n, 2 * n))

    aux_wire = total_qubits

    # prepare transformed state
    # state already contains auxiliary dimension
    qml.StatePrep(
        state,
        wires=range(total_qubits + 1),
        pad_with=0.0
    )

    # inverse order

    # undo y transform
    qml.adjoint(
        lambda: QChT([aux_wire] + y_wires)
    )()

    # undo x transform
    qml.adjoint(
        lambda: QChT([aux_wire] + x_wires)
    )()

    return qml.state()

def convert_deconvert_image_qcht_neqr(image_path, image_size=(128, 128)) -> np.ndarray:
    """

    :param image_path: path of image to be pipelined
    :param image_size: size of image. CORRESPONDS TO ORDER DEPENDING ON STATE (NEQR HERE)
    :return: Creates full pipeline of image -> neqr encoding -> qcht -> inverse qcht -> decoding -> reconstructed image.

    """

    path = image_path

    img = Image.open(path).convert("L")  # "L" = 8-bit grayscale
    img_resized = img.resize(image_size, Image.BICUBIC)

    # numpy array
    img_array_o = np.array(img_resized)

    # neqr image encoding
    state_neqr, b, neqr_x, neqr_y = timer(neqr_encode_image)(path)
    n_qubits_image_total_neqr = b + neqr_x + neqr_y  # + 1 auxillary is added in function

    transformed_image = timer(qcht_on_neqr_image)(state_neqr, b, neqr_x, neqr_y)
    # print(len(transformed_image))  # = 8388608 FOR 128 IMAGE, because 2^[aux (1) + 8 (bits) + 7x + 7y] = 8388608

    reconstructed_image = timer(inverse_qcht_image_neqr)(transformed_image, b, neqr_x, neqr_y)
    # extract only states where auxiliary qubit is |0⟩
    n_data_qubits = b + neqr_x + neqr_y
    reconstructed_image_traced = np.zeros(2 ** n_data_qubits, dtype=complex)

    for i in range(2 ** n_data_qubits):
        # index where auxiliary (last qubit) is 0
        reconstructed_image_traced[i] = reconstructed_image[i * 2]

    aux_1_slice = reconstructed_image[1::2]  # aux = |1⟩
    print(f"max amplitude in aux=|1⟩ slice: {np.max(np.abs(aux_1_slice)):.20e}")
    print(f"max amplitude in aux=|0⟩ slice: {np.max(np.abs(reconstructed_image[0::2])):.20e}")
    # renormalization
    reconstructed_image_traced = reconstructed_image_traced / np.linalg.norm(reconstructed_image_traced)
    reconstructed_image_neqr = timer(reconstruct_neqr_state)(reconstructed_image_traced, b, neqr_x, neqr_y)

    # back to array
    img_recon_neqr = np.array(reconstructed_image_neqr, dtype=np.float32)


    return img_recon_neqr


def convert_deconvert_image_qcht_nass(image_path, image_size=(128, 128)) -> np.ndarray:
    """

    :param image_path: path of image to be pipelined
    :param image_size: size of image. CORRESPONDS TO ORDER DEPENDING ON STATE (NASS HERE)
    :return: Creates full pipeline of image -> nass encoding -> qcht -> inverse qcht -> decoding -> reconstructed image.

    """

    nass_encoded_image, nx, ny, original_norm = timer(nass_encode_image)(image_path, size=image_size)

    transformed_image = timer(qcht_on_nass_image)(nass_encoded_image, nx, ny)
    detransformed_image = timer(inverse_qcht_image_nass)(transformed_image, nx, ny)

    n_data_qubits = nx + ny
    # filtering out auxilarry qubit
    detransformed_image_traced = np.zeros(
        2 ** n_data_qubits,
        dtype=complex
    )

    for i in range(2 ** n_data_qubits):
        # auxiliary qubit = |0>
        detransformed_image_traced[i] = detransformed_image[i * 2]

    detransformed_image_traced /= np.linalg.norm(
        detransformed_image_traced
    )

    reconstructed_state = timer(reconstruct_nass_state)(detransformed_image_traced, nx, ny, original_norm)

    return reconstructed_state


def convert_deconvert_image_qcht_frqi(image_path, image_size=(128, 128))-> np.ndarray:
    """
   :param image_path: path of image to be pipelined
   :param image_size: size of image. CORRESPONDS TO ORDER DEPENDING ON STATE (FRQI HERE)
   :return: Creates full pipeline of image -> frqi encoding -> qcht -> inverse qcht -> decoding -> reconstructed image.
    """

    frqi_encoded_image, n = timer(frqi_encode_image)(image_path, image_size)
    transformed_image = timer(qcht_on_frqi_image)(frqi_encoded_image, n)
    detransformed_image = timer(inverse_qcht_image_frqi)(transformed_image, n)


    n_pixels = 2 ** (2 * n)
    # --------------------------------------------------
    frqi_recovered = np.zeros(2 ** (2 * n + 1), dtype=complex)

    for i in range(2 ** (2 * n + 1)):
        # Keep only amplitudes where auxiliary qubit = |0>
        frqi_recovered[i] = detransformed_image[2 * i]

    # Renormalize
    frqi_recovered /= np.linalg.norm(frqi_recovered)

    # --------------------------------------------------
    # Decode FRQI
    # --------------------------------------------------
    scale = 2 ** n

    pixels = np.zeros(n_pixels)

    for i in range(n_pixels):
        a = frqi_recovered[2 * i] * scale  # cos(theta_i)
        b = frqi_recovered[2 * i + 1] * scale  # sin(theta_i)

        theta = np.arctan2(b.real, a.real)
        pixels[i] = theta * 255.0 / (np.pi / 2)

    reconstructed_frqi_image = pixels.reshape((2 ** n, 2 ** n))

    return reconstructed_frqi_image