# 1
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.io import wavfile

#
def DFT(x):
    m = np.arange(len(x))
    k = m.reshape((len(x), 1))
    F = np.exp(-2j * np.pi * k * m / len(x))
    return np.dot(F, x)

def FFT(x):
    if len(x) <= 1:
        return x
    x_par = FFT(x[::2])
    x_impar = FFT(x[1::2])
    fact = np.exp(-2j * np.pi * np.arange(len(x) // 2) / len(x))
    return np.concatenate([x_par + fact, x_par - fact])

N = [128, 256, 512, 1024, 2048, 4096, 8192]
t1, t2, t3 = [], [], []

for n in N:
    x = np.random.rand(n)
    t_init1 = time.time()
    dft = DFT(x)
    t_fin1 = time.time() - t_init1
    t1.append(t_fin1)

    t_init2 = time.time()
    fft = FFT(x)
    t_fin2 = time.time() - t_init2
    t2.append(t_fin2)

    t_init3 = time.time()
    fft_np = FFT(x)
    t_fin3 = time.time() - t_init3
    t3.append(t_fin3)

plt.figure(figsize=(10,6))
plt.suptitle('Exercitiul 1 - comparatie intre timpii de executie')
plt.stem(N, t1, markerfmt = 'go', label='DFT')
plt.stem(N, t2, markerfmt = 'C1x', label='FFT')
plt.stem(N, t3, label='FFT din numpy')
plt.xlabel('Dimensiune')
plt.ylabel('Timp')
plt.yscale('log')
plt.legend()
plt.savefig('ploturi/ex1.pdf', format = 'pdf')
plt.show()

# 2
t = np.linspace(0, 1, 1000)
te = np.linspace(0, 1, 9) # f esantionare = 9, fsemnal = 10
s = 2 * np.sin(2 * np.pi * 10 * te + np.pi/6)
semnal1 = 2 * np.sin(2 * np.pi * 10 * t + np.pi/6)
semnal2 = 2 * np.sin(2 * np.pi * 2 * t + np.pi/6)
semnal3 = 2 * np.sin(2 * np.pi * (10 - 2 * 9 + 2) * t + np.pi/6)
fig, axs = plt.subplots(4)
for ax in axs:
    ax.set(xlabel ='Timp', ylabel = 'Amplitudine')
fig.suptitle('Exercitiul 2 - aliere')
semnale = [semnal1, semnal2, semnal3]
axs[0].plot(t, semnal1)
i = 1
for ax in axs[1:]:
    ax.plot(te, s, marker = 'o', linestyle = 'None', markersize = 6)
    ax.plot(t, semnale[i - 1])
    i += 1
plt.savefig('ploturi/ex2.pdf', format = 'pdf')
plt.show()

# 3
te = np.linspace(0, 1, 20)
s = 2 * np.sin(2 * np.pi * 10 * te + np.pi/6)
t = np.linspace(0, 1, 1000) # f semnal = 10, f esantion = 21
semnal1 = 2 * np.sin(2 * np.pi * 10 * t + np.pi/6)
semnal2 = 2 * np.sin(2 * np.pi * 12 * t + np.pi/6)
semnal3 = 2 * np.sin(2 * np.pi * (10 -2 *(21 - 1)) * t + np.pi/6)
fig, axs = plt.subplots(4)
for ax in axs:
    ax.set(xlabel ='Timp', ylabel = 'Amplitudine')
fig.suptitle('Exercitiul 2 - fara aliere pentru frecvente mai mari decat frecventa Nyquist')
semnale = [semnal1, semnal2, semnal3]
axs[0].plot(t, semnal1)
i = 1
for ax in axs[1:]:
    ax.plot(te, s, marker = 'o', linestyle = 'None', markersize = 6)
    ax.plot(t, semnale[i - 1])
    i += 1
plt.savefig('ploturi/ex3.pdf', format = 'pdf')
plt.show()

# 4
# frecventa trebuie sa fie esantionata cu mai mult decat frecventa maxima
# fs > 2 * 200 = 400

# 5
# in ploturi/vocale.png se afla spectrograma generata pe baza
# vocalele se disting intre ele in mare parte (a nu se distinge foarte mult de e)

# 6
# a
frecv, semnal = wavfile.read('audio/vocale.wav')
semnal = np.array(semnal)
semnal = semnal.astype(float)
if semnal.ndim > 1:
    semnal = semnal.mean(axis=1)
N = len(semnal)
grup = int(0.01 * N)
suprapunere = grup // 2
pas = grup - suprapunere
nr_ferestre = (N - suprapunere) // pas
spectrograma = []
for i in range(nr_ferestre):
    inc = i * pas
    sf = inc + grup
    if sf > N:
        break
    fereastra = semnal[inc:sf]
    fft = np.fft.fft(fereastra)
    fft_val = np.abs(fft[:pas // 2])
    spectrograma.append(fft_val)

spectrograma = np.array(spectrograma).T
spectrograma = 10 * np.log10(spectrograma + 1e-6)

plt.imshow(spectrograma, aspect ='auto', cmap ='plasma')
plt.suptitle('Exercitiul 6 - spectrograma')
plt.savefig('ploturi/ex6.pdf', format = 'pdf')
plt.show()


# 7
# P_semnal = 90 db
# SNR_dB = 80 dB
# P_noise = ?
# SNR_dB = 10 * log(10) (P_semnal / P_noise)
# SNR_dB / 10 = log(10) (P_semnal / P_noise)
# 10 ** (SNR_dB / 10) = P_semnal / P_noise
# P_noise = P_semnal / 10 ** (SNR_dB / 10)
# P_noise = (10 ** (90 / 10)) / 10 ** (80 / 10) = (10 ** 9) / (10 ** 8) = 10
# P_noise = 10 dB


