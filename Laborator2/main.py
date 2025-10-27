import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile

# 1
def semnal_sin(amplitudine, frecv, t, faza):
    return amplitudine * ( np.sin(2 * np.pi * frecv * t + faza))
def semnal_cos(amplitudine, frecv, t, faza):
    return amplitudine * ( np.cos(2 * np.pi * frecv * t + faza))
t = np.linspace(0, 0.1, 200)

# sin(x) = cos(x - pi/2)
fig, axs = plt.subplots(2)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel = 'Amplitudine')
axs[0].plot(t, semnal_sin(2, 100, t, np.pi/3))
axs[1].plot(t, semnal_cos(2, 100, t, np.pi/3 - np.pi/2))
fig.suptitle("Exercitiul 1")
plt.savefig('ploturi/ex1.pdf', format = 'pdf')
plt.show()

# 2
t = np.linspace(0, 0.1, 400)
plt.plot(t, semnal_sin(1, 60, t, np.pi/13)) # albastru
plt.plot(t, semnal_sin(1, 60, t, np.pi/5)) # portocaliu
plt.plot(t, semnal_sin(1, 60, t, np.pi/7)) # verde
plt.plot(t, semnal_sin(1, 60, t, np.pi/3)) # rosu
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.suptitle("Exercitiul 2 - fara zgomot")
plt.savefig('ploturi/ex2faranoise.pdf', format = 'pdf')

# cu zgomot, ultimul semnal de faza pi/3
fig, axs = plt.subplots(4)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel = 'Amplitudine')
x = semnal_sin(1, 60, t, np.pi/3)
z = np.random.normal(size = len(t))
i = 0
for snr in [0.1, 1, 10, 100]:
    gamma = np.sqrt( np.linalg.norm(x) ** 2 / (snr * np.linalg.norm(z) ** 2))
    semnal = x + gamma * z
    axs[i].plot(t, semnal)
    i += 1
plt.suptitle("Exercitiul 2 - cu zgomot")
plt.savefig('ploturi/ex2cunoise.pdf', format = 'pdf')
plt.show()
# 3
fs = 44100
def semnal_square(frecv, t):
    return np.sign(np.sin(2 * frecv * t * np.pi))
def semnal_sawtooth(frecv, t):
    return 2 * (frecv * t - np.floor(t * frecv + 1/2))
# a
t = np.linspace(0, 3, fs)
semnal1 = semnal_sin(1, 400, t, 0)
sd.play(semnal1, 44100)
sd.wait()

# b
t = np.linspace(0, 3, fs)
semnal2 = semnal_sin(1, 800, t, 0)
sd.play(semnal2, 44100)
sd.wait()

# c
t = np.linspace(0, 3, fs)
semnal3 = semnal_sawtooth(240, t)
sd.play(semnal3, 44100)
sd.wait()

# d
t = np.linspace(0, 3, fs)
semnal4 = semnal_square(300, t)
sd.play(semnal4, 44100)
sd.wait()
wavfile.write('audio/sin_400.wav', 44100, semnal1.astype(np.float32))
wavfile.write('audio/sin_800.wav', 44100, semnal2.astype(np.float32))
wavfile.write('audio/sawtooth.wav', 44100, semnal3.astype(np.float32))
wavfile.write('audio/square.wav', 44100, semnal4.astype(np.float32))
frecv, semnal = wavfile.read('audio/square.wav')
sd.play(semnal, frecv)
sd.wait()

# 4
t = np.linspace(0, 0.1, 200)
fig, axs = plt.subplots(3)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel = 'Amplitudine')
axs[0].plot(t, semnal_sawtooth(100, t))
axs[1].plot(t, semnal_sin(1, 200, t, np.pi/3))
axs[2].plot(t, semnal_sawtooth(100, t) + semnal_sin(1, 200, t, np.pi/3))
fig.suptitle("Exercitiul 4")
plt.savefig('ploturi/ex4.pdf', format = 'pdf')

# 5
fs = 44100
t = np.linspace(0, 2, 2 * fs )
semnal1 = semnal_square(200, t)
semnal2 = semnal_square(300, t)
semnale = np.concatenate([semnal1, semnal2])
sd.play(semnale, fs)
sd.wait()

# 6
t = np.linspace(0, 0.1, 400)
fig, axs = plt.subplots(3)
fs = 200
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel = 'Amplitudine')
axs[0].plot(t, semnal_sin(1, fs / 2, t, 0))
axs[1].plot(t, semnal_sin(1, fs / 4, t, 0))
axs[2].plot(t, semnal_sin(1, 0, t, 0))
fig.suptitle("Exercitiul 6")
plt.savefig('ploturi/ex6.pdf', format = 'pdf')

# 7
# a
fig, axs = plt.subplots(3)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel = 'Amplitudine')
fs = 1000
t = np.linspace(0, 0.05, int(fs * 0.2))
semnal = semnal_sin(1, 100, t, 0)
t_dec_a = t[::4]
semnal_dec_a = semnal[::4]
axs[0].plot(t, semnal)
axs[1].stem(t_dec_a, semnal_dec_a)
# b
t_dec_b = t[1::4]
semnal_dec_b = semnal[1::4]
axs[2].stem(t_dec_b, semnal_dec_b)
plt.savefig('ploturi/ex7.pdf', format = 'pdf')
plt.show()

# 8
alfa = np.linspace(- np.pi/2, np.pi/2, 2000)
sinus = np.sin(alfa)
pade = (alfa -  7 * alfa ** 3 /60) / (1 + alfa ** 2 / 20)
fig, axs = plt.subplots(2)
axs[0].plot(alfa, alfa, color = 'green', label = 'Taylor')
axs[0].plot(alfa, sinus, color = 'purple', label = 'Sin')
axs[0].plot(alfa, pade, color = 'pink', label = 'Pade')
axs[0].legend()
axs[1].semilogy(alfa, np.abs(sinus - alfa), color = 'green', label = '|sin - Taylor|')
axs[1].semilogy(alfa, np.abs(sinus - pade), color = 'purple', label = '|sin - Pade|')
fig.suptitle("Exercitiul 8")
axs[1].legend()
plt.savefig('ploturi/ex8.pdf', format = 'pdf')

plt.show()