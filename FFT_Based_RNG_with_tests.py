import numpy as np, matplotlib.pyplot as plt
from scipy import signal
import hashlib

#Creating the Signal with two frequencies
dt = 0.001
t = np.arange(0, 300, dt)
f = signal.sawtooth(2*np.pi*100*t) + np.sin(2*np.pi*50*t)
f_clean = f 
f = f + 2.5*np.random.randn(len(t))  # Adding noise

n = len(t)
fhat = np.fft.fft(f, n)  
PSD = fhat * np.conj(fhat) / n  # Power Spectral Density 
freq = (1/(dt*n)) * np.arange(n)  # Frequency bins
L = np.arange(1, np.floor(n/2), dtype='int')  # Only positive frequencies

# Plotting the Power Spectral Density (PSD) of the noisy signal
fig,axs = plt.subplots(2, 1, figsize=(15, 11))
plt.sca(axs[0])
plt.plot(t,f, color = "red", linewidth = 1.5, label='Noisy Signal')
plt.plot(t, f_clean, color = "Black", linewidth = 2, label='Clean Signal')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color = "blue", linewidth = 1.5, label = "Noisy Signal")
plt.title ("Power Spectral Density (PSD) as function of Frequency")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.xlabel("Frequency (Hz)")
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()


# Using this to generate Bit Stream (0s and 1s)

threshold = np.median(PSD)  # Threshold for bit generation
bits = (PSD > threshold).astype(int)  # Generate bits based on threshold
half = len(bits) // 2
folded_bits = bits[:half] ^ bits[half:half*2] # Fold the bits

#Reducing the bias using Von Neumann Extraction 

def von_neumann_extraction(folded_bits):
    extracted_bits = []
    for i in range (0, len(folded_bits) - 1, 2):
        pairs = folded_bits[i], folded_bits[i + 1]
        if pairs == (0,1) or pairs == (1,0):
            extracted_bits.append(folded_bits[i])
    return np.array(extracted_bits)

#Randomness Extraction using SHA256 Cryptographic Hash Function

def bits_to_bytes(bits): 
    
    pad_len =(8 - len(bits) % 8) % 8  # Padding to make the length a multiple of 8 (It just calculates how many Zeros so that total number of bits is a multiple of 8)  
    bits_padded = np.concatenate((bits, np.zeros(pad_len, dtype=int))) # Adds those zeros to the end of the bits array
    bytes_arr = np.packbits(bits_padded)  # Converting bits to bytes
    return bytes_arr.tobytes() #Numpy array to Python bytes object so that SHA256 can process it.

def sha256_extractor(bits):
    block_size = 256 #Size of each block (256 bits = 32 bytes)
    final_bits = [] #Initializing the final bits array to store the output bits
    
    for i in range (0, len(bits), block_size):
        block = bits[i:i + block_size] #Looping through the bits in blocks of size 256 bits
        if len(block) < block_size: 
            break #If the block is < 256 bits, we stop processing further
        
        
        block_bytes = bits_to_bytes(block) #Convert the block of bits to bytes
        hash_bytes = hashlib.sha256(block_bytes).digest() #Apply SHA256 to the bytes and get the 256-bit hash 
        hash_bits = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8)) #Convert hash bytes to bits (0s and 1s)
        final_bits.extend(hash_bits[:block_size]) #Take all the above 256 bits and add them to our initialized final bits array
    
    return np.array(final_bits)

final_bits = sha256_extractor(von_neumann_extraction(folded_bits))
print("Final Bit Stream Length:", len(final_bits))


import math
import numpy as np
from scipy import special

# Convert final_bits to numpy array
bits = np.asarray(final_bits).astype(np.uint8)

# --- Monobit (Frequency) Test ---
def monobit_test(bits):
    n = len(bits)
    s = 2*np.sum(bits) - n  # #1 - #0
    s_obs = abs(s)/math.sqrt(n)
    p_value = special.erfc(s_obs/math.sqrt(2))
    return p_value

# --- Block Frequency Test ---
def block_frequency_test(bits, M=128):
    n = len(bits)
    if n < M:
        return np.nan, "INSUFFICIENT"
    Nblocks = n // M
    blocks = bits[:Nblocks*M].reshape((Nblocks, M))
    pi = np.mean(blocks, axis=1)
    chi2 = 4.0*M*np.sum((pi - 0.5)**2)
    p_value = special.gammaincc(Nblocks/2.0, chi2/2.0)
    return p_value, Nblocks

# --- Runs Test ---
def runs_test(bits):
    n = len(bits)
    pi = np.mean(bits)
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return 0.0, False  # Fails prerequisite
    V = 1 + np.sum(bits[1:] != bits[:-1])
    numerator = abs(V - 2*n*pi*(1-pi))
    denom = 2*math.sqrt(2*n)*pi*(1-pi)
    p_value = special.erfc(numerator/denom)
    return p_value, True

# --- FFT (Spectral) Test ---
def fft_spectral_test(bits):
    n = len(bits)
    X = 2*bits - 1
    import numpy.fft as fft
    S = np.abs(fft.fft(X))[:n//2]
    T = np.sqrt(math.log(1/0.05)*n)
    N0 = 0.95*(n/2.0)
    N1 = np.sum(S < T)
    d = (N1 - N0)/math.sqrt(n*0.95*0.05/4.0)
    p_value = special.erfc(abs(d)/math.sqrt(2))
    return p_value, N1, N0, T

# Run tests
alpha = 0.01
print("\\n=== NIST-Like Test Results (alpha = 0.01) ===")
p_monobit = monobit_test(bits)
print(f"Monobit Test: p = {p_monobit:.6f} -> {'PASS' if p_monobit >= alpha else 'FAIL'}")

p_block, Nblocks = block_frequency_test(bits, M=128)
if not np.isnan(p_block):
    print(f"Block Frequency Test (M=128, Nblocks={Nblocks}): p = {p_block:.6f} -> {'PASS' if p_block >= alpha else 'FAIL'}")
else:
    print("Block Frequency Test: INSUFFICIENT DATA")

p_runs, prereq_ok = runs_test(bits)
if prereq_ok:
    print(f"Runs Test: p = {p_runs:.6f} -> {'PASS' if p_runs >= alpha else 'FAIL'}")
else:
    print("Runs Test: FAIL (prerequisite not met)")

p_fft, N1, N0, T = fft_spectral_test(bits)
print(f"FFT Spectral Test: p = {p_fft:.6f} -> {'PASS' if p_fft >= alpha else 'FAIL'}")
