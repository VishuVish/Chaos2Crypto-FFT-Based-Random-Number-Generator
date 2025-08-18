import numpy as np, matplotlib.pyplot as plt
from scipy import signal
import hashlib

#Creating the Signal with two frequencies
dt = 0.001
t = np.arange(0, 15000, dt) 
f = 2.5*np.random.randn(len(t))  # Adding noise

n = len(t)
fhat = np.fft.fft(f, n)  
PSD = fhat * np.conj(fhat) / n  # Power Spectral Density 
freq = (1/(dt*n)) * np.arange(n)  # Frequency bins
L = np.arange(1, np.floor(n/2), dtype='int')  # Only positive frequencies

# Plotting the Power Spectral Density (PSD) of the noisy signal
fig,axs = plt.subplots(2, 1, figsize=(15, 11))
plt.sca(axs[0])
plt.plot(t,f, color = "red", linewidth = 1.5, label='Noisy Signal')
plt.title("Noisy Signal in Time-Domain") 
plt.xlim(t[0], t[-1])


plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color = "blue", linewidth = 1.5, label = "Noisy Signal")
plt.title ("Power Spectral Density (PSD) as function of Frequency")
plt.ylabel("Power/Frequency (dB/Hz)")
plt.xlabel("Frequency (Hz)")
plt.xlim(freq[L[0]], freq[L[-1]])

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

#Saving the final bits to a text file
with open("final_bits.txt", "w") as f:
    f.write(''.join(map(str, final_bits)))
