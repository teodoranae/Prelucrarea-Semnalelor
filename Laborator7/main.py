import pylab as pl
from IPython.core.pylabtools import figsize

from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets

# 1

# n1 = np.linspace(0, 1, 100)
# n2 = np.linspace(0, 1, 100)
# N1, N2 = np.meshgrid(n1, n2)
# f1 = np.sin(2 * np.pi * N1 + 3 * np.pi *N2)
# f2 = np.sin(4 * np.pi * N1) + np.cos(6 * np.pi * N2)
#
# X1 = np.fft.fft2(f1)
# freq1 = 20 * np.log10(np.abs(X1) + 1e-6)   # 2 puncte, corespunzatoare celor 2 frecvente dominante
#
# X2 = np.fft.fft2(f2)
# freq2 = 20 * np.log10(np.abs(X2) + 1e-6)
#
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# axs[0, 0].imshow(f1, cmap = 'gray')
# axs[0,0].set_title('f1')
# axs[0, 1].imshow(f2, cmap ='gray')
# axs[0,1].set_title('f2')
# axs[1, 0].imshow(freq1)
# axs[1,0].set_title('Spectru f1')
# axs[1, 1].imshow(freq2)
# axs[1,1].set_title('Spectru f2')
# plt.tight_layout()
# # plt.show()
# plt.savefig('ploturi/ex1_plot1.pdf', format = 'pdf')
#
# fig, axs = plt.subplots(3, 2, figsize=(10, 10))
# Y3 = np.zeros((100, 100), dtype = complex)
# Y3[0, 5] = Y3[0, 100 -5] = 1
# X3 = np.abs(np.fft.ifft2(Y3))
# freq3 = 20 * np.log10(np.abs(Y3) + 1e-6)
#
# Y4 = np.zeros((100, 100), dtype = complex)
# Y4[5, 0] = Y4[100 -5, 0] = 1
# X4 = np.abs(np.fft.ifft2(Y4))
# freq4 = 20 * np.log10(np.abs(Y4) + 1e-6)
#
# Y5 = np.zeros((100, 100), dtype = complex)
# Y5[5, 5] = Y5[100 -5, 100 - 5] = 1
# X5 = np.abs(np.fft.ifft2(Y5))
# freq5 = 20 * np.log10(np.abs(Y5) + 1e-6)
#
# axs[0, 0].imshow(X3)
# axs[0,0].set_title('f3')
# axs[0, 1].imshow(freq3)
# axs[0,1].set_title('Spectru f3')
# axs[1, 0].imshow(X4)
# axs[1,0].set_title('f4')
# axs[1, 1].imshow(freq4)
# axs[1,1].set_title('Spectru f4')
# axs[2, 0].imshow(X5)
# axs[2,0].set_title('f5')
# axs[2, 1].imshow(freq5)
# axs[2,1].set_title('Spectru f5')
#
# plt.tight_layout()
# # plt.show()
# plt.savefig('ploturi/ex1_plot2.pdf', format = 'pdf')


# 2
X = datasets.face(gray=True).astype(float)
# plt.imshow(X)
# plt.axis('off')
# plt.savefig('ploturi/original.pdf', format = 'pdf')
# Y = np.fft.fft2(X)
# freq = 20 * np.log10(abs(Y))
#
# snr = 0
# cutoff = 100
# i = 0
# fig, axs = plt.subplots(2, 2, figsize = (12, 10))
# while snr < 10:
#     Y_cutoff = Y.copy()
#     Y_cutoff[freq > cutoff] = 0
#     X_cutoff = np.fft.ifft2(Y_cutoff)
#     X_cutoff = np.real(X_cutoff)
#     eps = 1e-12
#     snr = 10 * np.log10(np.linalg.norm(X) ** 2 / (np.linalg.norm(X - X_cutoff) ** 2 + eps))
#     axs[i // 2, i % 2].imshow(X_cutoff, cmap='gray')
#     axs[i // 2, i % 2].set_title(f'SNR={snr:.2f}, cutoff={cutoff}')
#     i += 1
#     cutoff += 20
# plt.tight_layout()
# plt.show()
# plt.savefig('ploturi/ex2.pdf', format = 'pdf')

# 3
X = datasets.face(gray=True).astype(float)
zgomot = np.random.randint(-200, 201, size = X.shape)
X_zgomot = X + zgomot
snr1 = 10 * np.log10(np.linalg.norm(X) ** 2 / (np.linalg.norm(zgomot) ** 2))
Y_zgomot = np.fft.ifft2(X_zgomot)
freq = 20 * np.log10(abs(Y_zgomot))
cutoff = 120
Y_fara_zgomot = Y_zgomot.copy()
Y_fara_zgomot[freq > cutoff] = 0
X_fara_zgomot = np.real(np.fft.ifft2(Y_fara_zgomot))
snr2 = 10 * np.log10(np.linalg.norm(X) ** 2 / (np.linalg.norm(zgomot) ** 2 ))

plt.subplot(1,3,2)
plt.imshow(X_zgomot, cmap='gray')
plt.title(f'Noisy (SNR={snr1:.2f} dB)')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(X_fara_zgomot, cmap='gray')
plt.title(f'Filtered (SNR={snr2:.2f} dB)')
plt.axis('off')
plt.show()