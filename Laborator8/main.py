import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1000)
trend = 0.0002 * t ** 2 + 0.003 * t + 0.2
sezon = 50 * np.sin(2 * np.pi * t / 100) + 75 * np.sin(2 * np.pi * t / 75)
variatii = np.random.normal(0, 10, 1000)
serie = trend + sezon + variatii
fig, axs = plt.subplots(4)
axs[0].plot(t, trend)
axs[1].plot(t, sezon)
axs[2].plot(t, variatii)
axs[3].plot(t, serie)
plt.savefig('ploturi/ex1a.pdf', format = 'pdf')
plt.show()


# b
serie2 = serie -np.mean(serie)
rez = np.correlate(serie2, serie2, mode = 'full')
rez = rez / rez.max()
pct = np.arange(len(rez))
plt.stem(pct[:100], rez[:100])
plt.title("Vectorul de autocorelatie")
plt.savefig('ploturi/ex1b.pdf', format = 'pdf')
plt.show()

# c
p = 5
X, Y = [], []
for t_idx in range(p, 1000):
    X.append(serie[t_idx - p: t_idx][::-1])
    Y.append(serie[t_idx])
X, Y = np.array(X), np.array(Y)
coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
Y_pred = []
for i in range(p, 1000):
    Y_pred.append(np.dot(coef, serie[i - p: i][:: -1]))
Y_pred =  np.array(Y_pred)
plt.plot(serie, label = 'Serie originala', color = 'green')
plt.plot(np.arange(p, 1000), Y_pred, label = 'Predictii', color = 'red')
plt.legend()
plt.savefig('ploturi/ex1c.pdf', format = 'pdf')
plt.show()


# d
train, test = serie[:800], serie[800:]
err = []
best_mse, best_p = float('inf'), 0
def fit(serie, p):
    X, Y = [], []
    for t_idx in range(p, 800):
        X.append(serie[t_idx - p: t_idx][::-1])
        Y.append(serie[t_idx])
    X, Y = np.array(X), np.array(Y)
    coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return coef

def predict(serie, coef, p, st, sf):
    Y_pred = []
    for i in range(st, sf + 1):
        Y_pred.append(np.dot(coef, serie[i - p: i][:: -1]))
    Y_pred = np.array(Y_pred)
    return Y_pred

for p in range(1, 10):
    coef = fit(train, p)
    Y_pred = predict(serie, coef, p,800, 999)
    mse =  np.mean((test - Y_pred) ** 2)
    err.append(mse)
    if mse < best_mse:
        best_p, best_mse = p, mse
print(f"Cel mai bun p={best_p} cu MSE={best_mse:.4f}")
# Cel mai bun p=8 cu MSE=207.2383