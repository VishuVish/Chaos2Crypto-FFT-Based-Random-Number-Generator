# Random Bit Generator from Noisy Signal Simulation

## Overview  
This code aims to generate random bits by simulating a noisy analog signal composed of two combined frequencies (a sawtooth and a sine wave *(can be any other wave forms)*) with added noise. The signal is processed in the frequency domain to extract bits from its Power Spectral Density (PSD). To improve randomness quality, a multi-stage extraction pipeline is applied:

1. **Bit generation from PSD thresholding and folding** to reduce bias.  
2. **Von Neumann extraction** to remove residual bias and correlations.  
3. **Cryptographic hashing using SHA-256** to strengthen the bitstream and to remove any tiniest amount of residual correlations.

The output is a high-quality pseudo-random bit sequence suitable for cryptographic and simulation purposes.

---

## Key Highlights  
- Produces over 1.2 million random bits in a single run.  
- Thoroughly tested with the NIST SP 800-22 randomness test suite *([Random Bitstream Tester](https://mzsoltmolnar.github.io/random-bitstream-tester/))* and passed all tests.

---

## Potential Applications  
- Simulated RNG for quantum communication protocols using Qiskit.  
- Educational tool to demonstrate randomness extraction from noisy physical signals.  

---

## NIST Test Results Summary  
The generated bitstream length: **1,248,768 bits** **(time window: 0 to 10000s)** 

All the following tests from the NIST SP 800-22 suite were passed with p-values above the 0.01 threshold, indicating no statistically significant deviation from randomness:

- Frequency (Monobit) Test  
- Frequency Test within a Block  
- Runs Test  
- Test for the Longest Run of Ones in a Block  
- Binary Matrix Rank Test  
- Non-overlapping Template Matching Test  
- Overlapping Template Matching Test  
- Maurer’s “Universal Statistical” Test  
- Linear Complexity Test  
- Serial Test  
- Approximate Entropy Test  
- Cumulative Sums (Cusum) Test  
- Random Excursions Test  
- Random Excursions Variant Test  

The Runs Test, for example, returned a p-value of approximately 0.03, comfortably passing the randomness criterion.

---

## Disclaimer

Randomness testing is inherently statistical, so passing all tests does not guarantee absolute randomness, nor does failing a single test necessarily indicate a flawed generator. Results can vary depending on the input data size and source. Use this code and results as a solid starting point, but always evaluate randomness thoroughly for your specific application.

---

Feel free to **clone**, **modify**, and **extend** this project to suit your needs!
