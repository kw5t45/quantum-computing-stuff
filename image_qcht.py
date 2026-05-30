import pennylane as qml
import numpy as np


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
def qcht_on_position_registers(state, b, nx, ny):
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
    # Wires assignment
    color_wires = list(range(b))
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

