import heapq
import numpy as np
class Nod:
    def __init__(self, simbol = None, frecv = 0):
        self.simbol = simbol
        self.frecv = frecv
        self.st = None
        self.dr = None
    def __lt__(self, other):
        return self.frecv < other.frecv

def frecvente(data):
    frecv = {}
    for sim in data:
        if sim in frecv:
            frecv[sim] += 1
        else:
            frecv[sim] = 1
    return frecv

def constructie_arbore_huffman(data):
    freq = frecvente(data)
    heap = [Nod(simbol, frecv) for simbol, frecv in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = Nod(frecv=n1.frecv + n2.frecv)
        parent.st = n1
        parent.dr = n2
        heapq.heappush(heap, parent)

    return heap[0]


def generare_coduri(nod, prefix='', coduri=None):
    if coduri is None:
        coduri = {}
    if nod is None:
        return
    if nod.simbol is not None:
        coduri[nod.simbol] = prefix
        return coduri
    generare_coduri(nod.st, prefix + '0', coduri)
    generare_coduri(nod.dr, prefix + '1', coduri)
    return coduri

def codare_huffman(data):
    arb = constructie_arbore_huffman(data)
    coduri = generare_coduri(arb)
    codat = ''.join(coduri[simbol] for simbol in data)
    return codat, coduri

def decodare_huffman(codat, coduri):
    coduri_inversate = {v: k for k, v in coduri.items()}
    decodat = []
    buffer = ""

    for bit in codat:
        buffer += bit
        if buffer in coduri_inversate:
            decodat.append(coduri_inversate[buffer])
            buffer = ""
    return np.array(decodat, dtype=int)

