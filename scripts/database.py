import os
import shutil
import tempfile
import zipfile
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

# Arquivos incluídos em exportação ZIP / aceitos na importação por ZIP
ARQUIVOS_BANCO_EXPORT = ("faces_db.pkl", "encodings.npy", "nomes.npy")

# -------------------------------
# Converter legado numpy para DataFrame
# -------------------------------
def _legacy_numpy_para_dataframe(encodings_path, nomes_path):
    if not os.path.exists(encodings_path) or not os.path.exists(nomes_path):
        return pd.DataFrame(columns=["nome", "embedding", "modelo"])

    encodings_np = np.load(encodings_path, allow_pickle=True)
    nomes_np = np.load(nomes_path, allow_pickle=True)

    encodings_arr = np.array(encodings_np, dtype=object)
    nomes_arr = np.array(nomes_np, dtype=object)

    tamanho = min(len(encodings_arr), len(nomes_arr))
    if tamanho == 0:
        return pd.DataFrame(columns=["nome", "embedding", "modelo"])

    return pd.DataFrame({
        "nome": nomes_arr[:tamanho].astype(str),
        "embedding": [np.array(e, dtype=np.float32) for e in encodings_arr[:tamanho]],
        "modelo": [DEEPFACE_MODEL] * tamanho,
    })


def _converter_legacy_para_dataframe():
    return _legacy_numpy_para_dataframe(LEGACY_ENCODINGS_PATH, LEGACY_NAMES_PATH)


def _listar_arquivos_banco_existentes():
    return [
        name for name in ARQUIVOS_BANCO_EXPORT
        if os.path.isfile(os.path.join(DATA_DIR, name))
    ]


def _normalizar_dataframe_import(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["nome", "embedding", "modelo"])
    if "nome" not in df.columns or "embedding" not in df.columns:
        raise ValueError("Formato inválido: o banco precisa das colunas 'nome' e 'embedding'.")
    out = df.copy()
    out["nome"] = out["nome"].astype(str)
    out["embedding"] = [np.array(v, dtype=np.float32) for v in out["embedding"].tolist()]
    if "modelo" not in out.columns:
        out["modelo"] = DEEPFACE_MODEL
    return out


def _dataframe_de_pasta_temp(pasta):
    pkl = os.path.join(pasta, "faces_db.pkl")
    if os.path.isfile(pkl):
        return _normalizar_dataframe_import(pd.read_pickle(pkl))
    enc = os.path.join(pasta, "encodings.npy")
    nom = os.path.join(pasta, "nomes.npy")
    if os.path.isfile(enc) and os.path.isfile(nom):
        return _legacy_numpy_para_dataframe(enc, nom)
    raise ValueError(
        "Não foi encontrado faces_db.pkl nem o par encodings.npy/nomes.npy no arquivo."
    )


def banco_possui_registros():
    df = carregar_dataframe()
    return not df.empty


def exportar_banco_zip(caminho_zip):
    presentes = _listar_arquivos_banco_existentes()
    if not presentes:
        raise ValueError("Nenhum arquivo de banco em data/ para exportar.")
    with zipfile.ZipFile(caminho_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in presentes:
            caminho = os.path.join(DATA_DIR, name)
            zf.write(caminho, arcname=name)


def extrair_dataframe_importacao(caminho):
    caminho = os.path.normpath(caminho)
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".pkl":
        return _normalizar_dataframe_import(pd.read_pickle(caminho))
    if ext == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(caminho, "r") as zf:
                for m in zf.infolist():
                    base = os.path.basename(m.filename)
                    if not base or base not in ARQUIVOS_BANCO_EXPORT:
                        continue
                    dest = os.path.join(tmp, base)
                    with zf.open(m, "r") as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            return _dataframe_de_pasta_temp(tmp)
    raise ValueError("Use um arquivo .pkl (faces_db.pkl) ou .zip exportado pelo sistema.")


def _remover_arquivos_banco_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in ARQUIVOS_BANCO_EXPORT:
        p = os.path.join(DATA_DIR, name)
        if os.path.isfile(p):
            os.remove(p)


def _zip_contem_arquivo_banco(caminho_zip):
    with zipfile.ZipFile(caminho_zip, "r") as zf:
        for m in zf.infolist():
            if os.path.basename(m.filename) in ARQUIVOS_BANCO_EXPORT:
                return True
    return False


def substituir_banco_de_arquivo(caminho):
    caminho = os.path.normpath(caminho)
    ext = os.path.splitext(caminho)[1].lower()
    if ext == ".pkl":
        df = _normalizar_dataframe_import(pd.read_pickle(caminho))
        _remover_arquivos_banco_data()
        salvar_dataframe(df)
        return
    if ext == ".zip":
        if not _zip_contem_arquivo_banco(caminho):
            raise ValueError(
                "O ZIP não contém arquivos de banco reconhecidos (faces_db.pkl ou encodings.npy/nomes.npy)."
            )
        _remover_arquivos_banco_data()
        with zipfile.ZipFile(caminho, "r") as zf:
            for m in zf.infolist():
                base = os.path.basename(m.filename)
                if base not in ARQUIVOS_BANCO_EXPORT:
                    continue
                dest = os.path.join(DATA_DIR, base)
                with zf.open(m, "r") as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        carregar_dataframe()
        return
    raise ValueError("Use um arquivo .pkl ou .zip.")


def concatenar_banco_de_arquivo(caminho):
    df_imp = extrair_dataframe_importacao(caminho)
    if df_imp.empty:
        raise ValueError("O arquivo importado não contém registros.")
    df_atual = carregar_dataframe()
    df_merged = pd.concat([df_atual, df_imp], ignore_index=True)
    salvar_dataframe(df_merged)

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