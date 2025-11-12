import numpy as np
import matplotlib.pyplot as plt

def plot_phase_output(state):


    phases = np.angle(state)

    plt.bar(range(16), phases)
    plt.xlabel("Basis state |k>")
    plt.ylabel("Phase (radians)")
    plt.title("Phase distribution after QFT")
    plt.show()

    probabilities = np.abs(state) ** 2

    plt.bar(range(16), probabilities)
    plt.xlabel("Basis state |k>")
    plt.ylabel("Probability")
    plt.title("Probability distribution after QFT(|101>)")
    plt.show()