#Quantum_Entanglement

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.compiler import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate
import numpy as np

qr = QuantumRegister(2, name="qr")
cr = ClassicalRegister(2, name="cr")
qc = QuantumCircuit(qr, cr)

#Beam Splitter (Single Qubit Version)
t = 1/np.sqrt(2)  
bs_2x2 = np.array([[t,t], [t,-t]])  # 2x2 unitary matrix for a beam splitter
bs_gate_single = UnitaryGate(bs_2x2, label="BS_single") 

# Controlled-X gate (This is our basically the CNOT Gate)
cx_matrix = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])  # Controlled-x gate matrix
cx_gate = UnitaryGate(cx_matrix, label="CZ")  # Controlled-x gate as a quantum gate

# Both the qubits are in the state |0> 

""" Based on which Bell state you want to generate, you can build the respective matrix as an np.array and
implement it as a UnitaryGate using qc.append() method. Also you need provide required qubit flips as well."""

qc.append(bs_gate_single, [qr[0]])# Apply the beam splitter
qc.append(cx_gate, [qr[0], qr[1]]) # Apply the controlled-X gate
qc.measure(qr,cr)
print(qc)

# Use Aer simulator backend
sim_01 = AerSimulator()
qc_t = transpile(qc, sim_01)

# Execute the circuit with n shots, n equal or greater than 1000
job = sim_01.run(qc_t, shots=1000)
result = job.result()
counts = result.get_counts()
print("Amplitude of the obtained state:", counts)

# Plotting the results
labels_p = {'10': r'$|10\rangle$','01': r'$|01\rangle$','11': r'$|11\rangle$','00': r'$|00\rangle$'}
Single_Photon_counts = {labels_p.get(k, k): v for k, v in counts.items()}
x = list(counts.keys())
y = list(counts.values())
plot_histogram(Single_Photon_counts)
plt.xlabel("Measurement Outcome")
plt.ylabel("Counts")
plt.title("Quantum Entangled Bell State Measurement")
plt.show()