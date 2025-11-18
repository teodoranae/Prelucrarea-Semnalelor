import numpy as np
import matplotlib.pyplot as plt

# a
# frecventa de esantionare : semnalul este esantionat din ora in ora,
# deci frecventa = 1 / 3600 Hz

# b
x = np.genfromtxt('Train.csv', delimiter=',', skip_header=1, dtype=[int, str, float])
# print(len(x))
# sunt 18288 esantioane
# 18288 / 24 = 762 zile = aprox. 2 ani si o luna

# c
# frecventa maxima = fs / 2 = 1 / 7200

# d
masini = np.array([elem[2] for elem in x])
fft_masini = np.abs(np.fft.fft(masini)[:len(masini) // 2])
frecvente = 1 / 3600 * np.linspace(0, len(masini) // 2, len(masini) // 2) / len(masini)
plt.plot(frecvente, fft_masini)
plt.xlabel('Frecventa')
plt.ylabel('| X(f) |')
plt.suptitle('FFT')
plt.savefig('ploturi/ex1d.pdf', format = 'pdf')
plt.show()

# e
media = np.mean(masini)
if media != 0:
    print("DC eliminata")
else:
    print("Nu este DC")
masini -= media
fft_masini_fara_dc = np.abs(np.fft.fft(masini)[:len(masini) // 2])
frecvente = 1 / 3600 * np.linspace(0, len(masini) // 2, len(masini) // 2) / len(masini)
plt.plot(frecvente, fft_masini_fara_dc)
plt.xlabel('Frecventa')
plt.ylabel('| X(f) |')
plt.suptitle('FFT - fara componenta continua')
plt.savefig('ploturi/ex1e.pdf', format = 'pdf')
plt.show()

# f
print("Cele 4 frecvente principale:")
top4_index = np.argsort(fft_masini_fara_dc)[-4:][::-1]
plt.plot(frecvente, fft_masini_fara_dc)
plt.xlabel('Frecventa')
plt.ylabel('| X(f) |')
plt.suptitle('Primele 4 frecvente')
for i in top4_index:
    print(f'{i}: Frecventa: {frecvente[i]}')
    plt.scatter(frecvente[i], fft_masini_fara_dc[i])
plt.savefig('ploturi/ex1f.pdf', format = 'pdf')
plt.show()

# g
esantioane_lunare = 30 * 24 # = 720
# am ales 22 oct 2012, ora 00:00
# 1392
plt.plot(masini[1392: 1392 + esantioane_lunare])
plt.xlabel('Esantioane')
plt.ylabel('Masini')
plt.suptitle('O luna de trafic incepand cu 22 octombrie 2012')
plt.savefig('ploturi/ex1g.pdf', format = 'pdf')
plt.show()

# h
# Necunoscand data de inceput, dar stiind ca semnalul se esantioneaza o data pe ora, putem
# efectua o analiza lunara (30 * 24 esantioane) sau anuala (365 * 30 * 24 esantioane). Din analiza lunara, se
# pot deduce cu o oarecare usurinta zilele de weekend, avand o medie mai mica a traficului
# decat o zi lucratoare. Aceasta deductie se poate confirma sau infirma folosind analiza anuala.
# Neajunsuri posibile: se poate alege o perioada neprielnica pentru o astfel de analiza
# (o luna de vara, cand, datorita vacantei, traficul este mai redus decat in timpul anului scolar,
# o perioada fara sarbatori sau o perioada cand au loc multe evenimente care deviaza traficul in oras).
# Factori de care depinde acuratetea: contextul social  si cultural al zonei
# (organizarea evenimentelor), SNR, perioada aleasa pentru analiza, adica lungimea semnalului

# i
# esantionare odata la 12 ore (la jumatate de zi)
masini = np.array([elem[2] for elem in x])
masini -= np.mean(masini)
fft_masini = np.fft.fft(masini)
frecvente = 1 / 3600 * np.linspace(0, len(masini) // 2, len(masini) // 2) / len(masini)
frecv_noua_12h = 1 / (12 * 3600)
semnal_filtrat = fft_masini.copy()
semnal_filtrat[: len(masini) // 2][np.abs(frecvente) > frecv_noua_12h] = 0
semnal_filtrat[len(masini) // 2 :][np.abs(frecvente) > frecv_noua_12h] = 0
masini_filtrat = np.fft.ifft(semnal_filtrat).real
fig, axs = plt.subplots(2, 1)
axs[0].plot(masini)
axs[0].set_title('Semnal original')
axs[0].grid(True)
axs[1].plot(masini_filtrat)
axs[1].set_title('Semnal filtrat')
axs[1].grid(True)
plt.suptitle('Filtrarea semnalului')
plt.tight_layout()
plt.savefig('ploturi/ex1i.pdf', format = 'pdf')
plt.show()
