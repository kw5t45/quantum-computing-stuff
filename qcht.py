import matplotlib.pyplot as plt
import pennylane as qml
import numpy as np


# Number of qubits (non-auxiliary qubit).
N = 7


def CNOT_ladder():
    for wire in range(1, N + 1):
        qml.CNOT([0, wire])


pi = np.pi


def rotate_phases():
    """Rotates and shifts the phase of the auxiliary qubit and rotates the jth qubit by
    pi/2^(j+1) in Z."""
    qml.RZ(-pi * (2**N - 1) / 2 ** (N + 1), wires=0)
    qml.PhaseShift(-pi / 2 ** (N + 1), wires=0)
    for wire in range(1, N + 1):
        qml.RZ(pi / 2 ** (wire + 1), wires=wire)

def permute_elements():
    """Reorders amplitudes of the conditioned states."""
    for wire in reversed(range(1, N + 1)):
        control_wires = [0] + list(range(wire + 1, N + 1))
        qml.MultiControlledX(wires=(*control_wires, wire))

def adjust_phases():
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

def QChT():
    """Performs the quantum Chebyshev transform."""
    qml.Hadamard(wires=0)
    CNOT_ladder()
    qml.QFT(wires=range(N + 1))
    rotate_phases()
    permute_elements()
    CNOT_ladder()
    adjust_phases()


dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit(state=0):
    qml.BasisState(state=state, wires=range(1, N + 1))
    QChT()
    return qml.state()


def circuit_to_draw():
    qml.BasisState(state=0, wires=range(1, N + 1))
    QChT()




j = 7  # Initial state in computational basis.
# fig, ax = qml.draw_mpl(circuit_to_draw, decimals=2, style="pennylane")()
# fig.show()
# plt.savefig('seven_qubit_qcht.png')

total_state = circuit(state=j)  # AFTER QCHT, LENGTH SHOULD BE 2^N + 1
#print(f'State after QCHT:{total_state}')


#inv_state = Inverse_QChT(
#     state=j,
#     num_qubits=N,
#     plot=False,
#     filename="inverse_qcht_7q.png"
# )

@qml.qnode(dev)
def inverse_from_statevector(statevec):
    # Prepare the full state on N+1 qubits using StatePrep
    qml.StatePrep(statevec, wires=range(N + 1))

    # Apply inverse QChT
    qml.adjoint(QChT)()

    return qml.state()

inv_state = inverse_from_statevector(total_state)
print(inv_state)
################
original_state_index = j  # if auxiliary qubit starts at 0
max_ampl_index = np.argmax(np.abs(inv_state)**2)

print("Index of largest amplitude:", max_ampl_index)
print("Amplitude at that index:", inv_state[max_ampl_index])

#print(f'State after Inverse QCHT:{inv_state}')
