import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler

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
p = 5
X, Y = [], []
for t_idx in range(p, 1000):
    X.append(serie[t_idx - p: t_idx][::-1])
    Y.append(serie[t_idx])
X, Y = np.array(X), np.array(Y)
coef_ar = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
Y_pred = []
for i in range(p, 1000):
    Y_pred.append(np.dot(coef_ar, serie[i - p: i][:: -1]))
Y_pred =  np.array(Y_pred)
plt.plot(serie, label = 'Serie originala', color = 'green')
plt.plot(np.arange(p, 1000), Y_pred, label = 'Predictii', color = 'red')
plt.legend()
plt.savefig('ploturi/2.pdf', format = 'pdf')
plt.show()

# 3
p_max = 20
X, Y = [], []
for t_idx in range(p_max, len(serie)):
    X.append(serie[t_idx - p_max: t_idx][::-1])
    Y.append(serie[t_idx])
X = np.array(X)
Y = np.array(Y)

# la fiecare pas testam toti regresorii nefolositi si il adaugam pe cel care minimizeaza mse
selectate = []
ramase = list(range(p_max))
coef_greedy = np.zeros(p_max)
nivel_sparsitate = 6
best_j, best_coef = None, None

for pas in range(nivel_sparsitate):
    mse_min = np.inf
    best_j, best_coef = None, None
    for j in ramase:
        idx = selectate + [j]
        X_sub = X[:, idx]
        coef = np.dot(np.linalg.inv(np.dot(X_sub.T, X_sub)), np.dot(X_sub.T, Y))
        Y_pred = np.dot(X_sub, coef)
        mse = np.mean((Y - Y_pred) ** 2)

        if mse < mse_min:
            mse_min = mse
            best_j = j
            best_coef = coef
    selectate.append(best_j)
    ramase.remove(best_j)
for i, j in enumerate(selectate):
    coef_greedy[j] = best_coef[i]

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1,1)).flatten()

lambda_l1 = 3
p = X.shape[1]
P = np.block([[np.dot(X_scaled.T, X_scaled), np.zeros((p, p))], [np.zeros((p, p)), np.zeros((p, p))]])
q = np.hstack([np.dot(-X_scaled.T, Y_scaled), lambda_l1 * np.ones(p)])
G = np.block([[np.eye(p), -np.eye(p)], [-np.eye(p), -np.eye(p)], [ np.zeros((p, p)), -np.eye(p)]])
h = np.zeros(3 * p)
P = matrix(P)
q = matrix(q)
G = matrix(G)
h = matrix(h)
lags = np.arange(1, p_max + 1)
solvers.options['show_progress'] = False
solution = solvers.qp(P, q, G, h)
coef_cvx = np.array(solution['x'][:p]).flatten()


fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axs[0].stem(lags, coef_greedy, basefmt=" ")
axs[0].set_title('AR sparse – Greedy')
axs[0].set_xlabel('Lag')
axs[0].set_ylabel('Coeficient')
axs[0].grid(True)

axs[1].stem(lags, coef_cvx, basefmt=" ")
axs[1].set_title('AR sparse – CVXOPT l1')
axs[1].set_xlabel('Lag')
axs[1].set_ylabel('Coeficient')
axs[1].grid(True)
plt.tight_layout()
plt.savefig('ploturi/3.pdf', format='pdf')
plt.show()

# 4
def calcul_radacini(coef):
    coef = np.array(coef, dtype = complex)
    n = len(coef) - 1
    coef = coef / coef[0]
    C = np.zeros((n, n), dtype=complex)
    C[1:, :-1] = np.eye(n-1)
    C[:, -1] = -coef[1:][::-1]
    radacini = np.linalg.eigvals(C)
    return radacini

coef = [2, 3, 7, 5, 4, -6]
radacini = calcul_radacini(coef)
plt.figure(figsize=(6,6))
plt.scatter(radacini.real, radacini.imag, color='red', s=50)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.xlabel("Partea reala")
plt.ylabel("Partea imaginara")
plt.title("Radacinile polinomului in planul complex")
plt.grid(True)
plt.savefig('ploturi/4.pdf', format='pdf')
plt.show()

# 5
def verifica_stationaritate(model,coef):
    coef = np.array(coef, dtype = complex)
    polinom = np.concatenate([[1], -coef])
    radacini = calcul_radacini(coef)
    if np.all(np.abs(radacini) > 1):
        print(f'Model {model} stationar')
    else:
        print(f'Model {model} nestationar')
    return radacini

rad_ar = verifica_stationaritate('AR', coef_ar)
rad_greedy = verifica_stationaritate('Greedy', coef_greedy)
rad_l1 = verifica_stationaritate('l1', coef_cvx)
plt.figure(figsize=(7,7))

t = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(t), np.sin(t), 'k--', alpha=0.3, label='Cercul unitate')

plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.scatter(rad_ar.real, rad_ar.imag, color='red', s=80, label='AR')

plt.scatter(rad_greedy.real, rad_greedy.imag, color='blue', s=80, label='Greedy')

plt.scatter(rad_l1.real, rad_l1.imag, color='green', s=80, label='L1 (CVXOPT)')

plt.xlabel("Partea reala")
plt.ylabel("Partea imaginara")
plt.title("Radacinile polinomului pentru modelele AR")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.savefig('ploturi/5.pdf', format='pdf')
plt.show()
