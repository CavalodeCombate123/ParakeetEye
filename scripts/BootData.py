import face_recognition
import os
import numpy as np

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

        imagem = face_recognition.load_image_file(caminho_imagem)
        faces = face_recognition.face_encodings(imagem)

        if len(faces) == 1:
            encodings.append(faces[0])
            nomes.append(pessoa)

np.save(os.path.join(data_path, "encodings.npy"), encodings)
np.save(os.path.join(data_path, "nomes.npy"), nomes)

print("Banco de encodings criado.")