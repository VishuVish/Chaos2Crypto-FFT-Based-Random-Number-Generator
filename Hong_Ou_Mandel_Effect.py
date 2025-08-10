# Hong-Ou-Mandel Effect using Strawberry Fields

import strawberryfields as sf
from strawberryfields.ops import BSgate, Fock
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

prog = sf.Program(2)

with prog.context as q:
    Fock(1) | q[0]  # Photon in mode 0
    Fock(1) | q[1]  # Photon in mode 1
    BSgate() | (q[0], q[1])   # 50:50 Beam splitter

eng = sf.Engine("fock", backend_options={"cutoff_dim": 3}) 
result_qun = eng.run(prog)
state = result_qun.state
probs = state.all_fock_probs()

# Extracting the probabilities for the Fock states
p_20 = state.fock_prob([2, 0])
p_11 = state.fock_prob([1, 1])
p_02 = state.fock_prob([0, 2])

#plotting the obtained Probabilities
x = np.array([0, 1, 2]) 
y = np.array([p_02, p_11, p_20])
labels = ['|0>|2>', '|1>|1>', '|2>|0>']

#Gaussian Dip Fit
def gaussian_dip(x, baseline, amp):
    mean = 1
    stddev = 0.6
    return baseline - amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
popt, _ = curve_fit(gaussian_dip, x, y, p0=[max(y), max(y) - min(y)])

# Generating the dip fit
x_fit = np.linspace(0, 2, 100)
y_fit = gaussian_dip(x_fit, *popt)

#Plotting
plt.plot(x_fit, y_fit, '-', label='Gaussian Fit', color='red')
plt.xticks(x, labels)
plt.xlabel('Fock States')
plt.ylabel('Probability of measuring coincidences')
plt.title('Hong-Ou-Mandel Effect with (Gaussian Dip Fit)')
plt.show()