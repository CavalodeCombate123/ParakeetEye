import os
import numpy as np
import pandas as pd
from tkinter import messagebox
from constants import DEEPFACE_MODEL

# -------------------------------
# Caminhos de banco
# -------------------------------
DATA_DIR = "data"
DB_DF_PATH = os.path.join(DATA_DIR, "faces_db.pkl")
LEGACY_ENCODINGS_PATH = os.path.join(DATA_DIR, "encodings.npy")
LEGACY_NAMES_PATH = os.path.join(DATA_DIR, "nomes.npy")

# -------------------------------
# Converter legado numpy para DataFrame
# -------------------------------
def _converter_legacy_para_dataframe():
    if not os.path.exists(LEGACY_ENCODINGS_PATH) or not os.path.exists(LEGACY_NAMES_PATH):
        return pd.DataFrame(columns=["nome", "embedding"])

    encodings_np = np.load(LEGACY_ENCODINGS_PATH, allow_pickle=True)
    nomes_np = np.load(LEGACY_NAMES_PATH, allow_pickle=True)

    # Requisito: pegar dados numpy -> converter para array -> criar DataFrame pandas
    encodings_arr = np.array(encodings_np, dtype=object)
    nomes_arr = np.array(nomes_np, dtype=object)

    tamanho = min(len(encodings_arr), len(nomes_arr))
    if tamanho == 0:
        return pd.DataFrame(columns=["nome", "embedding"])

    df = pd.DataFrame({
        "nome": nomes_arr[:tamanho].astype(str),
        "embedding": [np.array(e, dtype=np.float32) for e in encodings_arr[:tamanho]],
        "modelo": [DEEPFACE_MODEL] * tamanho,
    })
    return df

# -------------------------------
# Carregar DataFrame do banco
# -------------------------------
def carregar_dataframe():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DB_DF_PATH):
        try:
            df = pd.read_pickle(DB_DF_PATH)
            if "nome" not in df.columns or "embedding" not in df.columns:
                return pd.DataFrame(columns=["nome", "embedding"])
            if "modelo" not in df.columns:
                df["modelo"] = DEEPFACE_MODEL
            return df
        except Exception:
            return pd.DataFrame(columns=["nome", "embedding"])

    # Migração automática do formato antigo para DataFrame
    df = _converter_legacy_para_dataframe()
    df.to_pickle(DB_DF_PATH)
    return df

# -------------------------------
# Salvar DataFrame no banco
# -------------------------------
def salvar_dataframe(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    if "modelo" not in df.columns:
        df["modelo"] = DEEPFACE_MODEL
    df.to_pickle(DB_DF_PATH)

# -------------------------------
# CARREGAR BANCO DE DADOS
# -------------------------------
def carregar_banco():
    try:
        df = carregar_dataframe()
        encodings = [np.array(v, dtype=np.float32) for v in df["embedding"].tolist()]
        nomes = df["nome"].astype(str).tolist()
        return encodings, nomes
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao carregar banco de dados:\n{e}")
        return [], []

# -------------------------------
# ADICIONAR PESSOA AO BANCO
# -------------------------------
def adicionar_pessoa(nome, novos_encodings):
    df = carregar_dataframe()
    novos_registros = pd.DataFrame({
        "nome": [nome] * len(novos_encodings),
        "embedding": [np.array(e, dtype=np.float32) for e in novos_encodings],
        "modelo": [DEEPFACE_MODEL] * len(novos_encodings),
    })
    df = pd.concat([df, novos_registros], ignore_index=True)
    salvar_dataframe(df)

# -------------------------------
# LISTAR NOMES DO BANCO
# -------------------------------
def listar_nomes():
    df = carregar_dataframe()
    if df.empty:
        return []
    return df["nome"].astype(str).tolist()

# -------------------------------
# DELETAR PESSOA DO BANCO
# -------------------------------
def deletar_pessoa(nome):
    df = carregar_dataframe()
    if df.empty:
        return
    df = df[df["nome"] != nome].reset_index(drop=True)
    salvar_dataframe(df)