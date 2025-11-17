# quantum-computing-stuff
#### Repository for testing and experimenting with Pennylane.
### Quantum Fourier Transform 

Assuming $n$ qubits, creating a superposition:
$\[
|\psi\rangle = \frac{1}{\sqrt{2^n}} \sum_{k=0}^{31} |ωk\rangle
\]$ 

With frequency $ω$ we can perform the QFT with `qft_superposition()` and visualize
the output on a plot using `plot_probability_distribution()` to plot probability amplitudes 
or `plot_phase_output()` to plot the angles / phases of the output after the QFT.

Below is an example of QFT acting on the above superposition for n=6 and ω=2:

![img](probability_distribution.png)

where the bars at 0 and 32 correspond to a frequency of 2 (spacing = 2^n / frequency).


