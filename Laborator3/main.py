import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1
fourier = np.zeros((8, 8), dtype = complex)
for m in range(8):
    for k in range(8):
        fourier[m][k] = np.exp(-2 * 1j * np.pi * m * k / 8)
fig, axs = plt.subplots(8)
for i in range(8):
    axs[i].plot(fourier[i].real)
    axs[i].plot(fourier[i].imag)
plt.savefig('ploturi/ex1.pdf', format = 'pdf')
plt.show()

print(np.linalg.norm(np.abs(np.matmul(fourier, fourier.conj().T)) - 8 * np.identity(8)))


# 2 - 1
t = np.linspace(0, 1, 1000)
semnal = np.sin(2 * np.pi * 5 * t)
fig, axs = plt.subplots(1, 2)
punct = 620
axs[0].axhline(y = 0)
axs[0].plot(t, semnal)
axs[0].plot(2 * [t[punct]], [0, semnal[punct]], color = 'red')
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('Amplitudine')
parte_reala = semnal * np.cos(-2 * np.pi * t)
parte_imaginara = semnal * np.sin(-2 * np.pi * t)
axs[1].plot(parte_reala, parte_imaginara)
axs[1].set_xlabel('Parte reala')
axs[1].set_ylabel('Parte imaginara')
axs[1].axhline(y = 0)
axs[1].axvline(x = 0)
plt.savefig('ploturi/ex2fig1.pdf', format = 'pdf')
culoare = np.sqrt(parte_reala ** 2 + parte_imaginara ** 2)
axs[1].scatter(parte_reala, parte_imaginara, c = culoare)
plt.tight_layout()
plt.show()

# 2 - 2
fig, axs = plt.subplots(2, 2)
t = np.linspace(0, 1, 1000)
i = 0
omega_val = [1, 2, 5, 7]
semnal = np.sin(2 * np.pi * 5 * t)
for ax in axs.flat:
    omega = omega_val[i]
    parte_reala = semnal * np.cos(-2 * np.pi * omega * t)
    parte_imaginara = semnal * np.sin(-2 * np.pi * omega * t)
    ax.axhline(y = 0)
    ax.axvline(x = 0)
    ax.set_xlabel('Parte reala')
    ax.set_ylabel('Parte imaginara')
    ax.title.set_text(f'Omega = {omega}')
    ax.plot(parte_reala, parte_imaginara)
    i += 1

plt.tight_layout()
plt.savefig('ploturi/ex2fig2.pdf', format = 'pdf')
plt.show()

# 3
t = np.linspace(0, 1, 2000)
semnal1 = 2 * np.cos(2 * np.pi * 50 * t)
semnal2 = 3 * np.cos(2 * np.pi * 25 * t)
semnal3 = 1.5 * np.cos(2 * np.pi * 5 * t)
semnal = semnal1 + semnal2 + semnal3
fig, axs = plt.subplots(2)
axs[0].set_xlabel('Timp')
axs[0].set_ylabel('x[n]')
axs[0].plot(t, semnal)
frecv = semnal.shape[0] / 400
X = np.zeros((400, 1), dtype = 'complex')
frecvente = np.array((range(400))) * frecv
i = 0
for frecventa in frecvente:
    X[i] = np.sum(semnal * np.exp(-2 * 1j * np.pi * frecventa * t))
    i += 1

axs[1].stem(frecvente[:20], np.abs(X)[:20])
axs[1].set_xlabel('Frecventa')
axs[0].set_ylabel('|X(omega)|')
plt.savefig('ploturi/ex3.pdf', format = 'pdf')
plt.show()
