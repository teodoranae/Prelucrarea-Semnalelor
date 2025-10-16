import numpy as np
import matplotlib.pyplot as plt
# 1
# a
t = np.linspace(0, 0.03, 60)
def semnal_cos(t, frecv, faza):
    return np.cos(2 * np.pi * frecv * t + faza)
# b
fig, axs = plt.subplots(3)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel ='Amplitudine')

axs[0].plot(t, semnal_cos(t, 260, np.pi/3))
axs[1].plot(t, semnal_cos(t, 140, - np.pi/3))
axs[2].plot(t, semnal_cos(t, 60, np.pi/3))
fig.suptitle('Exercitiul 1 - b')
plt.savefig('ploturi/ex1b.pdf', format = 'pdf')
plt.show()
# c
fig, axs = plt.subplots(3)
for ax in axs:
    ax.set(xlabel = 'Timp', ylabel ='Amplitudine')
# axs[0].plot(t, semnal_cos(t, 260, np.pi/3))
# axs[1].plot(t, semnal_cos(t, 140, - np.pi/3))
# axs[2].plot(t, semnal_cos(t, 60, np.pi/3))

t = np.linspace(0, 0.03, 7)

axs[0].stem(t, semnal_cos(t, 260, np.pi/3))
axs[1].stem(t, semnal_cos(t, 140, - np.pi/3))
axs[2].stem(t, semnal_cos(t, 60, np.pi/3))

fig.suptitle('Exercitiul 1 - c')
plt.savefig('ploturi/ex1c.pdf', format = 'pdf')
plt.show()


# 2
fig, axs = plt.subplots(4)
for ax in axs:
    ax.set(xlabel ='Timp', ylabel = 'Amplitudine')
def semnal_sin(t, frecv, faza):
    return np.sin(2 * np.pi * frecv * t + faza)
# a
t = np.linspace(0, 0.02, 1600)
axs[0].plot(t, semnal_sin(t, 400, 0))

# b
t = np.linspace(0, 3, 40000)
axs[1].plot(t[:int(0.004 * len(t))], semnal_sin(t[:int(0.004 * len(t))], 800, 0))

# c
t = np.linspace(0, 0.03, 200)
def semnal_sawtooth(frecv, t):
    return 2 * (frecv * t - np.floor(t * frecv + 1/2))
axs[2].plot(t, semnal_sawtooth(240, t))

# d
t = np.linspace(0, 0.02, 200)
def semnal_square(frecv, t):
    return np.sign(np.sin(2 * frecv * t * np.pi))
axs[3].plot(t, semnal_square(300, t))
plt.suptitle('Exercitiul 2 - a - d')
plt.savefig('ploturi/ex2abcd.pdf', format = 'pdf')
plt.show()

# e
semnal = np.random.rand(128, 128)
plt.imshow(semnal, cmap = 'gray')
plt.suptitle('Exercitiul 2 - e')
plt.xticks([])
plt.yticks([])
plt.savefig('ploturi/ex2e.pdf', format = 'pdf')

# f
def init_semnal():
    semnal = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            semnal[i][j] += np.cos(2 * np.pi/3 + (i+j))
    return semnal
semnal = init_semnal()
plt.imshow(semnal, cmap = 'gray')
plt.suptitle('Exercitiul 2 - f')
plt.xticks([])
plt.yticks([])
plt.savefig('ploturi/ex2f.pdf', format = 'pdf')


# 3
# a
# f = 1/T => T = 1/f = 1/ (2* 10**3) = 0.5 * 10**(-3) s = 0.5 ms
# b
# 4 biti = 1/2 B
# 2000 * 3600 * 1/2 = 36 * 10**5 B