import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, filtfilt

# 1
B = 1
t = np.linspace(-3, 3, 1000)
x = np.sinc(B * t) ** 2

plt.figure(figsize=(12, 8))
i = 1
for fs in [1, 1.5, 2, 4]:
    Ts = 1 / fs
    n_pos = int(3 / Ts)
    n_neg = - int(3 / Ts)
    n = np.arange(n_neg, n_pos + 1)
    t_esantion = n * Ts
    x_esantion = np.sinc(B * t_esantion) ** 2

    mat_T = t[:, None]
    mat_Tn = t_esantion[None, :]
    sinc_mat = np.sinc((mat_T - mat_Tn) / Ts)
    x_rec = np.dot(sinc_mat, x_esantion)

    ax = plt.subplot(2, 2, i)
    ax.plot(t, x, label='Original')
    ax.stem(t_esantion, x_esantion, linefmt='C1-', markerfmt='C1o', basefmt=' ')
    ax.plot(t, x_rec, 'g--')
    ax.set_title(f'Fs = {fs} Hz')
    ax.set_xlabel('t[s]')
    ax.set_ylabel('Amplitudine')
    ax.grid(True)
    ax.legend()
    i += 1

plt.tight_layout()
plt.savefig('ploturi/ex1.pdf', format='pdf')
plt.show()

# 2
N = 100
x = np.random.rand(N)
conv1 = np.convolve(x, x)
conv2 = np.convolve(conv1, conv1)
conv3 = np.convolve(conv2, conv2)
fig, axs = plt.subplots(4)
axs[0].plot(x)
axs[0].set_title('Original')
axs[1].plot(conv1)
axs[1].set_title('Prima convolutie')
axs[2].plot(conv2)
axs[2].set_title('A doua convolutie')
axs[3].plot(conv3)
axs[3].set_title('A treia convolutie')
plt.tight_layout()
plt.savefig('ploturi/ex2.pdf', format = 'pdf')
plt.show()
# cu fiecare iteratie, forma se apropie mai mult de distributia normala
# semnal bloc
x = np.zeros(N)
x[40:61] = 1
conv1 = np.convolve(x, x)
conv2 = np.convolve(conv1, conv1)
conv3 = np.convolve(conv2, conv2)
fig, axs = plt.subplots(4)
axs[0].plot(x)
axs[0].set_title('Original')
axs[1].plot(conv1)
axs[1].set_title('Bloc - Prima convolutie')
axs[2].plot(conv2)
axs[2].set_title('Bloc - A doua convolutie')
axs[3].plot(conv3)
axs[3].set_title('Bloc - A treia convolutie')
plt.tight_layout()
plt.savefig('ploturi/ex2_semnal_bloc.pdf', format = 'pdf')
plt.show()


# 3
p = np.random.randint(0, 100, size = N)
q = np.random.randint(0, 100, size = N)
r = np.convolve(p, q)
p_fft = np.fft.fft(p, 2 * N - 1)
q_fft = np.fft.fft(q, 2 * N - 1)
r_fft = np.real(np.fft.ifft(p_fft * q_fft))
fig, axs = plt.subplots(2)
axs[0].plot(r)
axs[0].set_title('Inmultirea polinoamelor')
axs[1].plot(r_fft)
axs[1].set_title('Convolutie cu FFT')
plt.savefig('ploturi/ex3.pdf', format = 'pdf')
plt.tight_layout()
plt.show()

# 4
t = np.linspace(0, 0.2, 20)
x = 2 * np.cos(5 * t * np.pi + np.pi/6)
d = 7
y = np.roll(x, d)

x_fft = np.fft.fft(x)
y_fft = np.fft.fft(y)

rez1 = np.fft.ifft(np.conj(x_fft) * y_fft)
rez2 = np.fft.ifft(y_fft / x_fft)

#  rezultatul primei metode - magnitudini
# print(f"Metoda 1:  {np.round(np.abs(rez1), 6)}")
#  rezultatul metodei 2 - magnitudini
# print(f"Metoda 2:  {np.round(np.abs(rez2), 6)}")

d1 = np.argmax(np.abs(rez1))
d2 = np.argmax(np.abs(rez2))
print(f"Deplasarea cu metoda 1: {d1}")
print(f"Deplasarea cu metoda 2: {d2}")

# 5
def fereastra_dreptunghiulara(n):
    return np.ones(n)
def fereastra_hanning(N):
    n = np.arange(N)
    return 0.5 * ( 1- np.cos(2 * np.pi * n / N))
t = np.linspace(0, 1, 1000) # am ales frecventa de esantionare = 1000
semnal_sin = 2 * np.sin(2* 100 * np.pi * t )
segment_semnal = semnal_sin[100:300]
segment_t = t[100:300]
cu_dreptunghi = segment_semnal * fereastra_dreptunghiulara(200)
cu_hanning = segment_semnal * fereastra_hanning(200)
fig, axs = plt.subplots(3)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel ='Amplitudine')
axs[0].plot(segment_t, segment_semnal)
axs[0].set_title('Semnal original')
axs[1].plot(segment_t, cu_dreptunghi)
axs[1].set_title('Semnal cu fereastra dreptunghiulara')
axs[2].plot(segment_t, cu_hanning)
axs[2].set_title('Semnal cu fereastra Hanning')
plt.tight_layout()
plt.savefig('ploturi/ex5.pdf', format = 'pdf')
plt.show()

# 6
# a
x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, usecols=2)
semnal = x[:72].astype(float)
t = np.arange(72)

# b
ferestre = [5, 9, 13, 17]
fig, axs = plt.subplots(5)
for ax in axs:
    ax.set(ylabel ='Vehicule')
axs[0].plot(t, semnal)
axs[0].set_title('Semnal original')
i = 1
for w in ferestre:
    filtrare = np.convolve(semnal, np.ones(w), 'valid') / w
    t_filtrare = np.arange(len(filtrare))
    axs[i].plot(t_filtrare, filtrare)
    axs[i].set_title(f'Medie alunecatoare cu w = {w}')
    i += 1
plt.tight_layout()
plt.savefig('ploturi/ex6b.pdf', format = 'pdf')
plt.show()

# c
# f_Nyquist = fs / 2
# se esantioneaza o data pe ora, deci fs = 1 esantion / h = 1/ (60 * 60),
# deci fs = 1 / 3600 Hz
# cutoff la 0.2
# alegem frecventa de taiere fc = fs * 0.2
# frecv normalizata = fc / f_Nyquist

# d
fs = 1 / 3600
fN = fs / 2
taiere = 0.2
fc = fs * taiere
frecv_norm = fc / fN

b_butterworth, a_butterworth = butter(N=5, Wn = frecv_norm, btype = 'low')
b_cheby, a_cheby = cheby1(N=5, Wn = frecv_norm, rp = 5, btype= 'low')
b_cheby2, a_cheby2 = cheby1(N=5, Wn = frecv_norm, rp = 7, btype= 'low')

# e
x_butter = filtfilt(b_butterworth, a_butterworth, semnal)
x_cheby = filtfilt(b_cheby, a_cheby, semnal)
x_cheby2 = filtfilt(b_cheby2, a_cheby2, semnal)
fig, axs = plt.subplots(4)
for ax in axs:
    ax.set(ylabel ='Vehicule')
axs[0].plot(t, semnal)
axs[0].set_title('Semnal original')
axs[1].plot(t, x_butter)
axs[2].plot(t, x_cheby)
axs[3].plot(t, x_cheby2)
plt.savefig('ploturi/ex6e.pdf', format = 'pdf')
plt.show()
# aleg Butterworth - raspuns neted in banda de tranzitie, atenueaza zgomotul de frecventa inalta

# f
ord = [3, 8]
rps = [2, 7]
fig, axs = plt.subplots(5)
for ax in axs:
    ax.set(ylabel ='Vehicule')
axs[0].plot(t, semnal)
axs[0].set_title('Semnal original')
i = 1
for N in ord:
    for rp in rps:
        b_butter, a_butter = butter(N, frecv_norm, btype='low', analog = False)
        b_cheby, a_cheby = cheby1(N, rp=rp, Wn=frecv_norm, btype='low', analog = False)
        x_butter = filtfilt(b_butter, a_butter, semnal)
        x_cheby = filtfilt(b_cheby, a_cheby, semnal)
        axs[i].set_title(f'Ordin = {N}, rp = {rp}')
        axs[i].plot(t, semnal)
        axs[i].plot(t, x_butter) # cu portocaliu e Butterworth
        axs[i].plot(t, x_cheby) # cu verde e Cebisev

        i += 1

plt.tight_layout()
plt.savefig('ploturi/ex6f.pdf', format = 'pdf')
plt.show()
