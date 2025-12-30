import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
from scipy import datasets
from huffman import *
import cv2 as cv

Q_jpeg = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]])


Q_ycbcr = np.array([[0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])

Q_rgb = np.array([[1, 0, 1.402],
                    [1, -0.344136, -0.714136],
                    [1, 1.772, 0]])

def rgb_ycbcr(X):
    offset = np.array([0, 128, 128])
    rez = np.dot(X, Q_ycbcr.T) + offset
    return np.clip(rez, 0, 255).astype(np.uint8)

def ycbcr_rgb(X):
    offset = np.array([0, 128, 128])
    rez = np.dot(X - offset, Q_rgb.T)
    return np.clip(rez, 0, 255).astype(np.uint8)


def compresie_jpeg(X, mse_max = 0):
    # video (nr_frameuri, h, w, c)
    video = len(X.shape) == 4
    if video:
        nr_frames = X.shape[0]
        X_rec_video = []
        for f in range(nr_frames):
            X_frame_rec = compresie_jpeg(X[f], mse_max)
            X_rec_video.append(X_frame_rec)
        return np.stack(X_rec_video)

    poza_color = len(X.shape) == 3
    # daca poza e color, trecem in YCbCr
    if poza_color:
        X = rgb_ycbcr(X)

    # aplicam padding, in caz ca dimensiunile pozei nu sunt multipli de 8
    h, w = X.shape[:2]
    h_pad = (h + 7) // 8 * 8
    w_pad = (w + 7) // 8 * 8
    if poza_color:
        X_pad = np.zeros((h_pad, w_pad, 3))
        X_pad[:h, :w, :] = X
    else:
        X_pad = np.zeros((h_pad, w_pad))
        X_pad[:h, :w] = X
    X_rec = np.copy(X_pad).astype(float)

    factor_compresie = 1
    while True:
        # 2d-dct pe blocuri de 8 x 8 + cuantizare
        X_dct = np.copy(X_pad).astype(float)
        for i in range(0, h_pad, 8):
            for j in range(0, w_pad, 8):
                if poza_color:
                    for canal in range(3):
                        bloc = X_pad[i: i + 8, j: j + 8, canal]
                        X_dct[i: i + 8, j: j + 8, canal] = np.round(dctn(bloc, norm='ortho') / (Q_jpeg * factor_compresie))
                else:
                    bloc = X_pad[i: i + 8, j: j + 8]
                    X_dct[i: i + 8, j: j + 8] = np.round(dctn(bloc, norm='ortho') / (Q_jpeg * factor_compresie))

        # codare Huffman
        if poza_color:
            coef_flat = np.concatenate([X_dct[:, :, canal].flatten().astype(int) for canal in range(3)])
        else:
            coef_flat = X_dct.flatten().astype(int)

        sir_biti, coduri = codare_huffman(coef_flat)

        # reconstructie
        # pasul 1 - decodare Huffman
        coef_decod = decodare_huffman(sir_biti, coduri)
        if poza_color:
            X_q_rec = np.stack([coef_decod[i * h_pad * w_pad: (i + 1) * h_pad * w_pad].reshape(h_pad, w_pad) for i in range(3)], axis = 2)
        else:
            X_q_rec = coef_decod.reshape(h_pad, w_pad)

        # idct
        for i in range(0, h_pad, 8):
            for j in range(0, w_pad, 8):
                if poza_color:
                    for canal in range(3):
                        bloc = X_q_rec[i: i + 8, j: j + 8, canal] * Q_jpeg * factor_compresie
                        X_rec[i: i + 8, j: j + 8, canal] = idctn(bloc, norm = 'ortho')
                else:
                    bloc = X_q_rec[i: i + 8, j: j + 8] * Q_jpeg * factor_compresie
                    X_rec[i: i + 8, j: j + 8] = idctn(bloc, norm='ortho')

        if poza_color:
            X_rec_fin = ycbcr_rgb(X_rec)
        else:
            X_rec_fin = X_rec.copy()
        # eliminare padding
        if poza_color:
            X_rec_fin = X_rec_fin[:h, :w, :]
        else:
            X_rec_fin = X_rec_fin[:h, :w]
        X_rec_fin = np.clip(X_rec_fin,0,255).astype(np.uint8)
        mse = np.mean((X.astype(float) - X_rec_fin.astype(float))**2)
        if mse_max == 0 or mse > mse_max:
            break
        factor_compresie += 0.3

    return X_rec_fin
