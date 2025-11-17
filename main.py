import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from plots import plot_phase_output, plot_probability_distribution
import math


def qft_rotations(wires):
    """ recursive applicaiton of controlled phase shift"""

    if len(wires) == 0:
        return

    # Apply Hadamard to the first qubit
    qml.Hadamard(wires=wires[0])

    # Apply controlled phase rotations
    for k in range(1, len(wires)):
        angle = np.pi / (2 ** k)
        qml.ControlledPhaseShift(angle, wires=[wires[k], wires[0]])

    # Recurse on the remaining qubits
    qft_rotations(wires[1:])


def swap_registers(wires):
    n = len(wires)
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])


def qft(wires):
    qft_rotations(wires)
    swap_registers(wires)


# setting up a device
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev)
def qft_circuit_4bit(binary_num: str):
    if binary_num[-1] == '1':
        qml.PauliX(wires=0)
    if binary_num[-2] == '1':
        qml.PauliX(wires=1)
    if binary_num[-3] == '1':
        qml.PauliX(wires=2)
    if binary_num[-4] == '1':
        qml.PauliX(wires=3)

    # Apply QFT
    qft(range(n_qubits))
    return qml.state()


@qml.qnode(dev)

def qft_superposition(n_qubits, state):

    qml.StatePrep(state, wires=range(n_qubits), normalize=True)

    qml.QFT(wires=range(n_qubits))

    return qml.state()


# qml.draw_mpl(qft_circuit(), decimals=2, style="sketch")()
# plt.show()


state = np.zeros(2 ** n_qubits, dtype=complex)
superposition_indexes = [i for i in range(0, 2**6, 2)]
for idx in superposition_indexes:
    state[idx] = 1 / math.sqrt(len(superposition_indexes))  # equal amplitude superposition of above list/

state = qft_superposition(6, state)
print(state)

# spacing -> 2^n/period
plot_probability_distribution(state, 6, save=True)
