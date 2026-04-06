import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import face_recognition
from tkinter.simpledialog import askstring
import os

from constants import *
from face_processing import *
from database import carregar_banco
from image_utils import mostrar_imagem_redimensionada

# -------------------------------
# Upload de imagem
# -------------------------------
def upload_imagem():
    encodings_conhecidos, nomes_conhecidos = carregar_banco()
    caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")])
    if not caminho:
        return

    try:
        imagem = face_recognition.load_image_file(caminho)
        imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)

        localizacoes = face_recognition.face_locations(imagem)
        encodings_imagem = face_recognition.face_encodings(imagem, localizacoes)

        resultados = []

        for (top, right, bottom, left), encoding in zip(localizacoes, encodings_imagem):
            ok_prof, _ = verificar_profundidade_face(imagem, (top, right, bottom, left))
            cor = (0, 255, 0)
            
            if not ok_prof:
                nome = "Possível foto/plano (não comparado ao banco)"
                cor = (0, 165, 255)
            elif len(encodings_conhecidos) == 0:
                nome = "Desconhecido"
            else:
                distancias = face_recognition.face_distance(encodings_conhecidos, encoding)
                if len(distancias) == 0:
                    nome = "Desconhecido"
                else:
                    melhor_match = np.argmin(distancias)
                    nome = nomes_conhecidos[melhor_match] if distancias[melhor_match] < 0.6 else "Desconhecido"

            resultados.append(nome)

            cv2.rectangle(imagem_bgr, (left, top), (right, bottom), cor, 2)
            cv2.putText(imagem_bgr, nome, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        if resultados:
            messagebox.showinfo("Resultado", "\n".join(resultados))
        else:
            messagebox.showinfo("Resultado", "Nenhum rosto detectado")

        mostrar_imagem_redimensionada(imagem_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao processar imagem:\n{e}")

# -------------------------------
# webcam
# -------------------------------
def abrir_webcam():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
            return

        frame_count = 0
        encodings_conhecidos, nomes_conhecidos = carregar_banco()
        
        # Lista para guardar detecções e não piscar a tela
        localizacoes = []
        nomes_detectados = []
        gray_pequeno_anterior = None

        while True:
            ok, frame = camera.read()
            if not ok: 
                break

            frame_count += 1
            
            # Otimização 
            if frame_count % 3 == 0:   # Melhoria: processa a cada 3 frames
                frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
                gray_pequeno = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                
                localizacoes = face_recognition.face_locations(rgb, model="hog")
                encodings_frame = face_recognition.face_encodings(rgb, localizacoes)
                
                nomes_detectados = []
                for loc, encoding in zip(localizacoes, encodings_frame):
                    top, right, bottom, left = loc
                    exigir_mov = gray_pequeno_anterior is not None
                    ok_prof, _ = verificar_profundidade_face(
                        rgb, loc, gray_anterior=gray_pequeno_anterior, exigir_movimento=exigir_mov
                    )
                    if not ok_prof:
                        nomes_detectados.append("Possível foto/plano")
                        continue
                        
                    if len(encodings_conhecidos) > 0:
                        distancias = face_recognition.face_distance(encodings_conhecidos, encoding)
                        melhor_match = np.argmin(distancias)
                        nome = nomes_conhecidos[melhor_match] if distancias[melhor_match] < 0.6 else "Desconhecido"
                    else:
                        nome = "Desconhecido"
                    nomes_detectados.append(nome)

                gray_pequeno_anterior = gray_pequeno.copy()

            # Desenhar
            for (top, right, bottom, left), nome in zip(localizacoes, nomes_detectados):
                top *= 4; right *= 4; bottom *= 4; left *= 4 
                cor = (0, 165, 255) if "Possível foto" in nome else (0, 255, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), cor, 2)
                cv2.putText(frame, nome, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) == 27: 
                break

        camera.release()
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Erro", f"Erro na webcam:\n{e}")

#-------------------------------
# CadastroPessoa
#-------------------------------
def cadastrar_pessoa():
    nome = askstring("Cadastro", "Digite o nome da pessoa:")

    if not nome or nome.strip() == "":
        messagebox.showwarning("Erro", "Nome inválido.")
        return

    nome = nome.strip().title()
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        messagebox.showerror("Erro", "Não foi possível abrir a câmera.")
        return

    novos_encodings = []
    contador = 0
    gray_anterior = None
    liberado_captura = False
    aviso_multiplo = ""

    while contador < 20:
        ok, frame = camera.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        ok_unico, msg_unico = validar_unico_rosto_para_cadastro(faces)
        aviso_multiplo = "" if ok_unico else msg_unico

        # Verificação de profundidade / vivacidade  
        if ok_unico:
            loc = faces[0]
            ok_prof, msg_prof = verificar_profundidade_face(
                rgb, loc, gray_anterior=gray_anterior, exigir_movimento=False
            )
            cinza = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mov = movimento_medio_roi(cinza, gray_anterior, *loc) if gray_anterior is not None else None
            if mov is not None and mov >= LIMIAR_MOVIMENTO_ROI:
                liberado_captura = True
            gray_anterior = cinza.copy()

            encodings = face_recognition.face_encodings(rgb, [loc]) if ok_prof else []

            if ok_prof and liberado_captura and encodings:
                encoding = encodings[0]
                if len(novos_encodings) == 0 or np.mean(face_recognition.face_distance([novos_encodings[-1]], encoding)) > 0.35:
                    novos_encodings.append(encoding)
                    contador += 1
            elif not ok_prof:
                aviso_multiplo = msg_prof
        else:
            gray_anterior = None

        cv2.putText(frame, f"{contador}/20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if aviso_multiplo:
            cv2.putText(frame, aviso_multiplo[:70], (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            
        if not liberado_captura and ok_unico:
            cv2.putText(frame, "Mova levemente o rosto (anti-foto)", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        cv2.imshow("Cadastro", frame)

        if cv2.waitKey(1) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

    if not liberado_captura:
        messagebox.showerror(
            "Cadastro",
            "Não foi detectada vivacidade (movimento natural do rosto). Segure uma foto parada não é aceito. Tente de novo.",
        )
        return

    if len(novos_encodings) < 8:   # Exige mínimo de amostras boas
        messagebox.showerror("Erro", "Poucas amostras válidas capturadas. Tente novamente.")
        return

    # carregar banco
    encodings_existentes, nomes_existentes = carregar_banco()

    # adicionar novos
    encodings_existentes.extend(novos_encodings)
    nomes_existentes.extend([nome] * len(novos_encodings))

    # salvar
    try:
        np.save("data/encodings.npy", encodings_existentes)
        np.save("data/nomes.npy", nomes_existentes)
        messagebox.showinfo("Sucesso", f"{nome} cadastrado com sucesso! ({len(novos_encodings)} amostras)")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao salvar:\n{e}")

#-------------------------------
# Mostrar pessoas cadastradas
#-------------------------------
def listar_pessoas(janela):  # parâmetro adicionado para modularidade
    try:
        nomes = np.load("data/nomes.npy", allow_pickle=True).tolist()
    except:
        messagebox.showinfo("Banco", "Nenhuma pessoa cadastrada.")
        return

    if not nomes:
        messagebox.showinfo("Banco", "Nenhuma pessoa cadastrada.")
        return

    # contar quantas amostras por pessoa
    contagem = {}
    for nome in nomes:
        contagem[nome] = contagem.get(nome, 0) + 1

    # criar nova janela
    janela_lista = tk.Toplevel(janela)
    janela_lista.title("Pessoas Cadastradas")
    janela_lista.geometry("450x500")

    # frame com scroll 
    frame = tk.Frame(janela_lista)
    frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)

    frame_lista = tk.Frame(canvas)

    frame_lista.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=frame_lista, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # função interna de deletar
    def deletar(nome):
        if not messagebox.askyesno("Confirmar", f"Tem certeza que deseja deletar {nome}?"):
            return

        try:
            nomes_np = np.load("data/nomes.npy", allow_pickle=True).tolist()
            encodings_np = np.load("data/encodings.npy", allow_pickle=True).tolist()

            novos_nomes = []
            novos_encodings = []

            for n, e in zip(nomes_np, encodings_np):
                if n != nome:
                    novos_nomes.append(n)
                    novos_encodings.append(e)

            np.save("data/nomes.npy", novos_nomes)
            np.save("data/encodings.npy", novos_encodings)

            messagebox.showinfo("Sucesso", f"{nome} removido.")
            janela_lista.destroy()
            listar_pessoas(janela)  # recarrega lista
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao deletar:\n{e}")

    # criar lista visual
    for nome, qtd in contagem.items():
        linha = tk.Frame(frame_lista)
        linha.pack(fill="x", pady=5, padx=5)

        tk.Label(linha, text=f"{nome} ({qtd})", anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(linha, text="❌", fg="red", command=lambda n=nome: deletar(n)).pack(side="right")