import numpy as np
import matplotlib.pyplot as plt

# # 1
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
alfa = 0.1
def mediere_exp(alfa, serie):
    rez = np.zeros_like(serie)
    rez[0] = serie[0]
    for t in range(1, len(serie)):
        rez[t] = alfa * serie[t - 1] + (1 - alfa) * rez[t - 1]
    return rez

def mse(y_true, y_pred):
    return np.mean((y_true[1:] - y_pred[1:]) ** 2)

alfas = np.linspace(0.01, 0.99, 99)
err = []
for a in alfas:
    rezultat = mediere_exp(a, serie)
    err.append(mse(serie, rezultat))
alfa_opt = alfas[np.argmin(err)]

def mediere_dubla(alfa, beta, serie):
    s = np.zeros_like(serie)
    b = np.zeros_like(serie)
    s[0] = serie[0]
    b[0] = serie[1] - serie[0]
    for t in range(1, len(serie)):
        s[t] = alfa * serie[t] + (1 - alfa) * (s[t - 1] + b[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * b[t- 1]
    return s + b

alfas = np.linspace(0.01, 0.99, 10)
betas = np.linspace(0.01, 0.99, 10)

best_err = np.inf
best_params = None

for alfa in alfas:
    for beta in betas:
            err = mse(serie, mediere_dubla(alfa, beta, serie))
            if err < best_err:
                best_err = err
                best_params = (alfa, beta)

def mediere_tripla(alfa, beta, gamma, serie):
    s = np.zeros_like(serie)
    b = np.zeros_like(serie)
    c = np.zeros_like(serie)

    s[0] = serie[0]
    b[0] = serie[1] - serie[0]
    c[0] = 0

    for t in range(1, len(serie)):
        s[t] = alfa * serie[t] + (1 - alfa) * (s[t - 1] + b[t - 1] + 0.5 * c[t - 1])
        b[t] = beta * (s[t] - s[t - 1]) + (1 - beta) * (b[t-1] + c[t-1])
        c[t] = gamma * (b[t] - b[t - 1]) + (1-gamma) * c[t-1]

    return s + b + c

alfas = np.linspace(0.01, 0.99, 10)
betas = np.linspace(0.01, 0.99, 10)
gammas = np.linspace(0.01, 0.99, 10)

best_err = np.inf
best_params = None

for alfa in alfas:
    for beta in betas:
        for gamma in gammas:
            err = mse(serie, mediere_tripla(alfa, beta, gamma, serie))
            if err < best_err:
                best_err = err
                best_params = (alfa, beta, gamma)

med_opt = mediere_exp(alfa_opt, serie)
dubla_opt = mediere_dubla(best_params[0], best_params[1], serie)
tripla_opt = mediere_tripla(best_params[0], best_params[1], best_params[2], serie)
fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs[0, 0].plot(serie, label="Seria originala", color="black")
axs[0, 0].set_title("Seria originala")
axs[0, 0].legend()
axs[0, 0].grid(True)

axs[0, 1].plot(serie, alpha=0.4, label="Originala")
axs[0, 1].plot(med_opt, label="Mediere exponentiala simpla (alfa optim)")
axs[0, 1].set_title("Mediere exponentiala simpla")
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 0].plot(serie, alpha=0.4, label="Originala")
axs[1, 0].plot(dubla_opt, label="Mediere exponentiala dubla (alfa, beta optim)")
axs[1, 0].set_title("Mediere exponentiala dubla")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].plot(serie, alpha=0.4, label="Originala")
axs[1, 1].plot(tripla_opt, label="Mediere exponentiala tripla (alfa, beta, gamma optim)")
axs[1, 1].set_title("Mediere exponentiala tripla")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('ploturi/2.pdf', format = 'pdf')
plt.show()

# 3
miu = np.mean(serie)
epsilon = serie - miu
def ma(serie, p):
    X = np.zeros((len(serie) - p, p))
    Y = np.zeros(len(serie) - p)
    for t in range(p, len(serie)):
        Y[t - p] = serie[t] - miu
        for j in range(p):
            X[t - p, j] = epsilon[t - j - 1]
    theta = np.linalg.lstsq(X, Y)[0]
    y_pred = np.zeros_like(serie)
    for i in range(p, len(serie)):
        val = miu
        for j in range(p):
            val += theta[j] * epsilon[i - j - 1]
        y_pred[i] = val
    return y_pred
p = 5
y_pred = ma(serie, p)

plt.figure(figsize=(12, 5))
plt.plot(serie, alpha=0.5, label="Serie originala")
plt.plot(y_pred, label=f"MA({p})")
plt.title("Model MA(p)")
plt.legend()
plt.grid(True)
plt.savefig('ploturi/3.pdf', format='pdf')
plt.show()

# 4
mse_min = np.inf
best_p, best_q, max_lag, best_pred = None, None, None, None
miu = np.mean(serie)
epsilon = serie - miu

def arma(serie, p, q):
    maxi = max(p, q)
    X = np.array([[serie[t - i] for i in range(1, p + 1)] +
                  [epsilon[t - j] for j in range(1, q + 1)]
                  for t in range(maxi, len(serie))])
    Y = np.array([serie[t] for t in range(maxi, len(serie))])
    params = np.linalg.lstsq(X, Y)[0]
    y_pred = X @ params
    mse = np.mean((Y - y_pred) ** 2)
    return y_pred, mse

for p in range(1, 21):
    for q in range(1, 21):
        y_pred, mse = arma(serie, p, q)
        if mse < mse_min:
            mse_min = mse
            best_p = p
            best_q = q
            best_pred = y_pred
            max_lag = max(p, q)

plt.figure(figsize=(12, 5))
max_lag = max(best_p, best_q)
pred = np.zeros_like(serie)
pred[:max_lag] = np.nan
pred[max_lag:] = y_pred
plt.plot(serie, 'k--', alpha=0.3, label='Serie originala')
plt.plot(pred, 'r', label=f'ARMA - predictii')
plt.title(f'ARMA cu p si q optime: p = {best_p}, q = {best_q}')
plt.legend()
plt.grid(True)
plt.savefig('ploturi/4_arma.pdf', format='pdf')
plt.show()

from statsmodels.tsa.arima.model import ARIMA

p, d, q = 2, 1, 2
model = ARIMA(serie, order=(p, d, q))
model_fit = model.fit()
y_pred = model_fit.predict(start=0, end=len(serie)-1)
plt.figure(figsize=(12, 5))
plt.plot(serie, 'k--', alpha=0.3, label='Serie originala')
plt.plot(y_pred, 'r', label=f'ARIMA')
plt.title(f'ARIMA cu p = {p}, d = {d}, q = {q}')
plt.legend()
plt.grid(True)
plt.savefig('ploturi/4_arima.pdf', format='pdf')
plt.show()
