import numpy as np
import matplotlib.pyplot as plt

def plot_phase_output(state, n_qubits, save=False):


    phases = np.angle(state)

    plt.bar(range(2 ** n_qubits), phases)
    plt.xlabel("Basis state |k>")
    plt.ylabel("Phase (radians)")
    plt.title("Phase distribution after QFT")

    if save:
        plt.savefig('phase_distrubtion.png')
    plt.show()

def plot_probability_distribution(state, n_qubits, save=False):

    probabilities = np.abs(state) ** 2

    plt.bar(range(2 ** n_qubits), probabilities)
    plt.xlabel("Basis state |k>")
    plt.ylabel("Probability")
    plt.title("Probability distribution after QFT")

    if save:
        plt.savefig('probability_distribution.png')
    plt.show()

