import os
import numpy as np
import cv2

from database import salvar_dataframe
# Dataset em lote: pode usar detector pesado (não impacta FPS da webcam)
from constants import DEEPFACE_DETECTOR_PESADO
from face_processing import detectar_faces, gerar_embedding
import pandas as pd

dataset_path = "dataset"
data_path = "data"
os.makedirs(data_path, exist_ok=True)

encodings = []
nomes = []

if not os.path.exists(dataset_path):
    print("Pasta 'dataset' não encontrada.")
    print("Criando banco vazio...")

    np.save(os.path.join(data_path, "encodings.npy"), np.array([], dtype=object))
    np.save(os.path.join(data_path, "nomes.npy"), np.array([], dtype=object))
    salvar_dataframe(pd.DataFrame(columns=["nome", "embedding", "modelo"]))

    print("Banco vazio criado.")
    exit()

# -------------------------------
# processamento normal
# -------------------------------
for pessoa in os.listdir(dataset_path):
    caminho_pessoa = os.path.join(dataset_path, pessoa)

    if not os.path.isdir(caminho_pessoa):
        continue

    for arquivo in os.listdir(caminho_pessoa):
        caminho_imagem = os.path.join(caminho_pessoa, arquivo)
        imagem_bgr = cv2.imread(caminho_imagem)
        if imagem_bgr is None:
            continue

        faces = detectar_faces(
            imagem_bgr,
            anti_spoofing=False,
            detector_backend=DEEPFACE_DETECTOR_PESADO,
        )
        if len(faces) != 1:
            continue

        encoding = gerar_embedding(imagem_bgr, faces[0]["loc"])
        if encoding is None:
            continue
        encodings.append(encoding)
        nomes.append(pessoa)

np.save(os.path.join(data_path, "encodings.npy"), encodings)
np.save(os.path.join(data_path, "nomes.npy"), nomes)
df = pd.DataFrame({"nome": nomes, "embedding": [np.array(e, dtype=np.float32) for e in encodings]})
salvar_dataframe(df)

print(f"Banco criado com {len(encodings)} embeddings.")