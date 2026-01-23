import numpy as np
import matplotlib.pyplot as plt

# 1
t = np.arange(1000)
comp1 = 0.0002 * t ** 2 + 0.003 * t + 0.2
comp2 = 50 * np.sin(2 * np.pi * t / 100) + 75 * np.sin(2 * np.pi * t / 75)
comp3 = np.random.normal(0, 10, 1000)
serie = comp1 + comp2 + comp3
fig, axs = plt.subplots(4)
axs[0].plot(t, comp1)
axs[1].plot(t, comp2)
axs[2].plot(t, comp3)
axs[3].plot(t, serie)
plt.tight_layout()
plt.savefig('ploturi/1.pdf', format='pdf')
plt.show()

# 2
L = 100
N = len(serie)
K = N - L + 1
X = np.zeros((L, K))
for i in range(L):
    X[i, :] = serie[i : i + K]
print(X)

# 3
XXT = X @ X.T
XTX = X.T @ X
eigenvals1, eigenvecs1 = np.linalg.eigh(XXT)
eigenvals2, eigenvecs2 = np.linalg.eigh(XTX)
idx = np.argsort(eigenvals1)[::-1]
eigenvals1_ord = eigenvals1[idx]
eigenvecs1_ord = eigenvecs1[:, idx]
idx = np.argsort(eigenvals2)[::-1]
eigenvals2_ord = eigenvals2[idx]
eigenvecs2_ord = eigenvecs2[:, idx]

U, S, V = np.linalg.svd(X)
S2 = S**2
print("Valori proprii XXT = S^2")
print(np.allclose(eigenvals1_ord[:10], S2[:10], atol=1e-8))
print("Valori proprii XTX = S^2")
print(np.allclose(eigenvals2_ord[:10], S2[:10], atol=1e-8))
print("Vectori proprii XXT = coloanele din U")
print(np.allclose(np.abs(eigenvecs1_ord[:, :10]),np.abs(U[:, :10]),atol=1e-8))
VT = V.T
print("Vectori proprii XTX = coloanele din V")
print(np.allclose(np.abs(eigenvecs2_ord[:, :10]),np.abs(VT[:, :10]), atol=1e-8))
# valorile proprii nenule ale matricelor XXT și XTX sunt aceleasi
# si sunt egale cu patratele valorilor singulare din SVD
# vectorii proprii ai lui XXT sunt coloanele din U
# vectorii proprii ai lui XTX sunt coloanele din V


# 4
def matrice_hankel(x, L):
    N = len(x)
    K = N - L + 1
    X = np.zeros((L, K))
    for i in range(L):
        X[i, :] = x[i:i+K]
    return X

L = 100
X = matrice_hankel(serie, L)
U, S, V = np.linalg.svd(X)
def matrice_elementara(U, S, VT, i):
    return S[i] * np.outer(U[:, i], VT[i, :])
r = 3
Xr = sum(matrice_elementara(U, S, V, i) for i in range(r))

def hankelizare(X):
    L, K = X.shape
    N = L + K - 1
    x_rec = np.zeros(N)
    cts = np.zeros(N)
    for i in range(L):
        for j in range(K):
            x_rec[i + j] += X[i, j]
            cts[i + j] += 1

    return x_rec / cts

componente = []
for i in range(L):
    Xi = matrice_elementara(U, S, V, i)
    xi = hankelizare(Xi)
    componente.append(xi)

serie_reconstruita = np.sum(componente, axis=0)
plt.figure(figsize=(10,4))
plt.plot(t, serie, label="Seria originala", alpha=0.6)
plt.plot(t, serie_reconstruita, '--', label="Reconstruita SSA", linewidth=2)
plt.title("Reconstructia seriei prin SSA")
plt.legend()
plt.grid()
plt.savefig('ploturi/4.pdf', format='pdf')
plt.show()




