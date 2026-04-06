import os
import numpy as np
from tkinter import messagebox

# -------------------------------
# CARREGAR BANCO DE DADOS
# -------------------------------
def carregar_banco():
    try:
        if not os.path.exists("data/encodings.npy") or not os.path.exists("data/nomes.npy"):
            os.makedirs("data", exist_ok=True)
            np.save("data/encodings.npy", np.array([], dtype=object))
            np.save("data/nomes.npy", np.array([], dtype=object))

        encodings = np.load("data/encodings.npy", allow_pickle=True).tolist()
        nomes = np.load("data/nomes.npy", allow_pickle=True).tolist()
        return encodings, nomes
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao carregar banco de dados:\n{e}")
        return [], []