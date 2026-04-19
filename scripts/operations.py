import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askstring

from constants import *
from face_processing import *
from database import (
    banco_possui_registros,
    carregar_banco,
    carregar_dataframe,
    adicionar_pessoa,
    concatenar_banco_de_arquivo,
    deletar_pessoa,
    exportar_banco_zip,
    extrair_dataframe_importacao,
    listar_nomes,
    substituir_banco_de_arquivo,
)
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
        imagem_bgr = cv2.imread(caminho)
        if imagem_bgr is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return
        imagem = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

        faces_detectadas = detectar_faces(
            imagem_bgr,
            anti_spoofing=False,
            detector_backend=DEEPFACE_DETECTOR_PESADO,
        )
        localizacoes = [face["loc"] for face in faces_detectadas]
        encodings_imagem = [gerar_embedding(imagem_bgr, loc) for loc in localizacoes]

        resultados = []

        anti_spoof_faces = detectar_faces(
            imagem_bgr,
            anti_spoofing=True,
            detector_backend=DEEPFACE_DETECTOR_PESADO,
        )

        for idx, ((top, right, bottom, left), encoding) in enumerate(zip(localizacoes, encodings_imagem)):
            anti_spoofing_info = anti_spoof_faces[idx] if idx < len(anti_spoof_faces) else None
            ok_prof, _ = verificar_profundidade_face(
                imagem,
                (top, right, bottom, left),
                imagem_bgr=imagem_bgr,
                anti_spoofing_info=anti_spoofing_info,
            )
            cor = (0, 255, 0)
            
            if encoding is None:
                nome = "Face inválida para embedding"
                cor = (0, 0, 255)
            elif not ok_prof:
                nome = "Possível foto/plano (não comparado ao banco)"
                cor = (0, 165, 255)
            elif len(encodings_conhecidos) == 0:
                nome = "Desconhecido"
            else:
                distancias = distancia_embeddings(encodings_conhecidos, encoding)
                if len(distancias) == 0:
                    nome = "Desconhecido"
                else:
                    melhor_match = np.argmin(distancias)
                    nome = nomes_conhecidos[melhor_match] if distancias[melhor_match] < LIMIAR_RECONHECIMENTO else "Desconhecido"

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
        # Reduz backlog de frames para minimizar "travadinhas" por frame antigo.
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        encodings_conhecidos, nomes_conhecidos = carregar_banco()
        tracks = {}
        next_track_id = 1
        frame_count = 0
        gray_anterior = None

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame_count += 1

            # Redimensiona para performance
            frame_small = cv2.resize(frame, (0, 0), fx=WEBCAM_SCALE, fy=WEBCAM_SCALE)
            rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            gray_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)

            # === Detecção de faces (leve e esparsa) ===
            if frame_count % WEBCAM_DETECT_EVERY == 0:
                detected = detectar_faces(
                    frame_small,
                    anti_spoofing=False,
                    detector_backend=DEEPFACE_DETECTOR_TEMPO_REAL,
                )
                locs = [f["loc"] for f in detected]
                matched, _ = associar_tracks_robusto(tracks, locs)

                new_tracks = {}
                for i, loc in enumerate(locs):
                    track_id = matched.get(i)
                    if track_id is None:
                        track_id = next_track_id
                        next_track_id += 1
                    new_tracks[track_id] = tracks.get(track_id, {
                        "loc": loc,
                        "nome": "Desconhecido",
                        "embedding": None,
                        "spoof_ok": True,
                        "spoof_fails": 0,
                        "last_embed": 0,
                        "last_spoof": 0
                    })
                    new_tracks[track_id]["loc"] = loc
                tracks = new_tracks

            # === Atualização de reconhecimento e anti-spoofing ===
            tracks_items = list(tracks.items())

            # Anti-spoofing pesado opcional: roda no quadro inteiro e em baixa frequência.
            spoof_faces_cache = []
            if WEBCAM_USAR_ANTISPOOF_PESADO and frame_count % WEBCAM_ANTISPOOF_EVERY == 0:
                spoof_faces_cache = detectar_faces(
                    frame_small,
                    anti_spoofing=True,
                    detector_backend=DEEPFACE_DETECTOR_PESADO,
                )

            # Limita embeddings por frame para evitar picos de processamento.
            candidatos_embed = []
            for track_id, data in tracks_items:
                if (
                    encodings_conhecidos
                    and data.get("spoof_ok", True)
                    and (frame_count - data["last_embed"] >= WEBCAM_EMBED_EVERY)
                ):
                    candidatos_embed.append((data["last_embed"], track_id))

            candidatos_embed = sorted(candidatos_embed, key=lambda x: x[0])
            permitidos_embed = {tid for _, tid in candidatos_embed[:WEBCAM_MAX_EMBEDS_PER_FRAME]}

            for track_id, data in tracks_items:
                loc = data["loc"]
                if "spoof_fails" not in data:
                    data["spoof_fails"] = 0

                # Anti-spoofing/liveness antes do reconhecimento.
                spoof_interval = WEBCAM_ANTISPOOF_EVERY if WEBCAM_USAR_ANTISPOOF_PESADO else WEBCAM_LIVENESS_EVERY
                if frame_count - data["last_spoof"] >= spoof_interval:
                    if WEBCAM_USAR_ANTISPOOF_PESADO and spoof_faces_cache:
                        top, right, bottom, left = loc
                        melhor = None
                        menor_delta = 10**9
                        for face in spoof_faces_cache:
                            t, r, b, l = face["loc"]
                            delta = abs(t - top) + abs(r - right) + abs(b - bottom) + abs(l - left)
                            if delta < menor_delta:
                                menor_delta = delta
                                melhor = face
                        ok_prof, _ = verificar_profundidade_face(
                            rgb_small,
                            loc,
                            gray_anterior=gray_anterior,
                            exigir_movimento=True,
                            imagem_bgr=frame_small,
                            anti_spoofing_info=melhor,
                        )
                    else:
                        # Modo tempo real: usa vivacidade por movimento sem chamar anti-spoofing pesado.
                        ok_prof, _ = verificar_profundidade_face(
                            rgb_small,
                            loc,
                            gray_anterior=gray_anterior,
                            exigir_movimento=True,
                            imagem_bgr=frame_small,
                            anti_spoofing_info={"is_real": True, "antispoof_score": 1.0},
                        )
                    if ok_prof:
                        data["spoof_fails"] = 0
                        data["spoof_ok"] = True
                    else:
                        data["spoof_fails"] += 1
                        data["spoof_ok"] = data["spoof_fails"] < WEBCAM_SPOOF_FAIL_TOLERANCE
                    data["last_spoof"] = frame_count

                # Embedding (reconhecimento) só se passou em vivacidade/anti-spoof recente
                if (
                    encodings_conhecidos
                    and data.get("spoof_ok", True)
                    and (frame_count - data["last_embed"] >= WEBCAM_EMBED_EVERY)
                    and track_id in permitidos_embed
                ):
                    emb = gerar_embedding(frame_small, loc)
                    if emb is not None:
                        data["embedding"] = emb
                        dists = distancia_embeddings(encodings_conhecidos, emb)
                        if len(dists) > 0:
                            best = np.argmin(dists)
                            data["nome"] = (
                                nomes_conhecidos[best]
                                if dists[best] < LIMIAR_RECONHECIMENTO_WEBCAM
                                else "Desconhecido"
                            )
                        data["last_embed"] = frame_count
                elif not data.get("spoof_ok", True):
                    # Mantém nome genérico enquanto suspeita de foto/plano
                    data["nome"] = "Possível foto/plano"

            gray_anterior = gray_small.copy()

            # === Desenho final ===
            for data in tracks.values():
                t, r, b, l = data["loc"]
                scale = 1 / WEBCAM_SCALE
                x1, y1, x2, y2 = int(l*scale), int(t*scale), int(r*scale), int(b*scale)

                nome = data.get("nome", "Desconhecido")
                cor = (0, 165, 255) if not data.get("spoof_ok", True) else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, nome, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2)

            cv2.imshow("ParakeetEye - Webcam", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
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
    frame_count = 0
    anti_spoof_cache_ok = None
    anti_spoof_cache_msg = "Aguardando anti-spoofing..."
    anti_spoof_countdown = 0
    loc_rosto_cache = None

    while contador < 30:
        ok, frame = camera.read()
        if not ok:
            break
        frame_count += 1

        frame = cv2.resize(frame, (0, 0), fx=WEBCAM_SCALE, fy=WEBCAM_SCALE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_count % CADASTRO_DETECT_EVERY == 0:
            faces_detectadas = detectar_faces(
                frame,
                anti_spoofing=False,
                detector_backend=DEEPFACE_DETECTOR_TEMPO_REAL,
            )
            faces = [face["loc"] for face in faces_detectadas]
            if len(faces) == 1:
                loc_rosto_cache = faces[0]
            else:
                loc_rosto_cache = None
        else:
            faces = [loc_rosto_cache] if loc_rosto_cache is not None else []
        ok_unico, msg_unico = validar_unico_rosto_para_cadastro(faces)
        aviso_multiplo = "" if ok_unico else msg_unico

        # Verificação de profundidade / vivacidade  
        if ok_unico:
            loc = faces[0]
            # Anti-spoofing pesado em frequência reduzida para evitar travamento.
            if USAR_ANTISPOOF_CADASTRO and anti_spoof_countdown <= 0:
                anti_spoof_faces = detectar_faces(
                    frame,
                    anti_spoofing=True,
                    detector_backend=DEEPFACE_DETECTOR_PESADO,
                )
                anti_spoofing_info = anti_spoof_faces[0] if anti_spoof_faces else None
                anti_spoof_cache_ok, anti_spoof_cache_msg = verificar_profundidade_face(
                    rgb,
                    loc,
                    gray_anterior=gray_anterior,
                    exigir_movimento=False,
                    imagem_bgr=frame,
                    anti_spoofing_info=anti_spoofing_info,
                )
                anti_spoof_countdown = 12
            else:
                anti_spoof_countdown -= 1

            if USAR_ANTISPOOF_CADASTRO:
                ok_prof = bool(anti_spoof_cache_ok)
                msg_prof = anti_spoof_cache_msg
            else:
                ok_prof = True
                msg_prof = ""

            cinza = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            mov = movimento_medio_roi(cinza, gray_anterior, *loc) if gray_anterior is not None else None
            if mov is not None and mov >= LIMIAR_MOVIMENTO_ROI:
                liberado_captura = True
            gray_anterior = cinza.copy()

            # Embedding ArcFace: menos frequente para manter FPS no cadastro
            deve_gerar_embedding = frame_count % 3 == 0
            encoding = gerar_embedding(frame, loc) if (ok_prof and deve_gerar_embedding) else None

            if ok_prof and liberado_captura and encoding is not None:
                if len(novos_encodings) == 0 or np.mean(distancia_embeddings([novos_encodings[-1]], encoding)) > LIMIAR_DIVERSIDADE_CADASTRO:
                    novos_encodings.append(encoding)
                    contador += 1
            elif not ok_prof:
                aviso_multiplo = msg_prof
        else:
            gray_anterior = None
            anti_spoof_countdown = 0
            loc_rosto_cache = None

        cv2.putText(frame, f"{contador}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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

    try:
        adicionar_pessoa(nome, novos_encodings)
        messagebox.showinfo("Sucesso", f"{nome} cadastrado com sucesso! ({len(novos_encodings)} amostras)")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao salvar:\n{e}")

#-------------------------------
# Mostrar pessoas cadastradas
#-------------------------------
def listar_pessoas(janela):  # parâmetro adicionado para modularidade
    try:
        nomes = listar_nomes()
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
            deletar_pessoa(nome)

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

#-------------------------------
# Exportar banco para Excel/CSV
#-------------------------------
def exportar_banco_planilha():
    try:
        df = carregar_dataframe()
    except Exception as e:
        messagebox.showerror("Exportação", f"Erro ao carregar banco:\n{e}")
        return

    if df.empty:
        messagebox.showinfo("Exportação", "Banco vazio. Não há dados para exportar.")
        return

    df_export = df.copy()
    if "nome" not in df_export.columns:
        messagebox.showerror("Exportação", "Formato do banco inválido (coluna 'nome' ausente).")
        return

    # Evita exportar vetor completo do embedding para planilha ficar legível.
    if "embedding" in df_export.columns:
        df_export["embedding_dim"] = df_export["embedding"].apply(
            lambda x: len(x) if hasattr(x, "__len__") else 0
        )
        df_export = df_export.drop(columns=["embedding"])

    caminho = filedialog.asksaveasfilename(
        title="Salvar planilha do banco",
        defaultextension=".xlsx",
        filetypes=[
            ("Excel", "*.xlsx"),
            ("CSV", "*.csv"),
        ],
        initialfile="faces_db_resumo.xlsx",
    )
    if not caminho:
        return

    try:
        if caminho.lower().endswith(".csv"):
            df_export.to_csv(caminho, index=False, encoding="utf-8-sig")
        else:
            try:
                df_export.to_excel(caminho, index=False)
            except Exception:
                # fallback para CSV quando não houver engine Excel (ex.: openpyxl ausente)
                caminho_csv = caminho.rsplit(".", 1)[0] + ".csv"
                df_export.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
                messagebox.showwarning(
                    "Exportação",
                    "Não foi possível gerar .xlsx neste ambiente. "
                    f"Arquivo CSV gerado em:\n{caminho_csv}",
                )
                return

        messagebox.showinfo(
            "Exportação",
            f"Planilha exportada com sucesso!\nRegistros: {len(df_export)}\nArquivo: {caminho}",
        )
    except Exception as e:
        messagebox.showerror("Exportação", f"Falha ao exportar planilha:\n{e}")


# -------------------------------
# Exportar arquivos do banco (ZIP)
# -------------------------------
def exportar_banco_arquivos_zip():
    try:
        caminho = filedialog.asksaveasfilename(
            title="Salvar backup do banco de dados",
            defaultextension=".zip",
            filetypes=[("Arquivo ZIP", "*.zip")],
            initialfile="parakeeteye_banco.zip",
        )
        if not caminho:
            return
        exportar_banco_zip(caminho)
        messagebox.showinfo(
            "Exportação",
            f"Arquivos do banco exportados com sucesso.\nArquivo:\n{caminho}",
        )
    except ValueError as e:
        messagebox.showinfo("Exportação", str(e))
    except Exception as e:
        messagebox.showerror("Exportação", f"Falha ao exportar ZIP:\n{e}")


# -------------------------------
# Menu: Exportar banco de dados
# -------------------------------
def exportar_banco_menu(janela_pai):
    topo = tk.Toplevel(janela_pai)
    topo.title("Exportar banco de dados")
    topo.geometry("360x150")
    topo.transient(janela_pai)
    topo.grab_set()

    def planilha():
        topo.destroy()
        exportar_banco_planilha()

    def zip_backup():
        topo.destroy()
        exportar_banco_arquivos_zip()

    tk.Label(topo, text="Escolha o tipo de exportação:").pack(pady=(14, 10))
    tk.Button(
        topo,
        text="Exportar planilha em Excel/CSV",
        command=planilha,
        width=32,
    ).pack(pady=4)
    tk.Button(
        topo,
        text="Exportar arquivos do banco (ZIP)",
        command=zip_backup,
        width=32,
    ).pack(pady=4)


# -------------------------------
# Importar banco de dados
# -------------------------------
def _pergunta_substituir_concatenar(janela_pai):
    escolha = [None]

    def definir(valor):
        escolha[0] = valor
        topo.destroy()

    topo = tk.Toplevel(janela_pai)
    topo.title("Importar banco de dados")
    topo.geometry("440x170")
    topo.transient(janela_pai)
    topo.grab_set()

    tk.Label(
        topo,
        text=(
            "Já existe um banco com dados em execução.\n"
            "Deseja substituir ou concatenar com o banco importado?"
        ),
        justify="center",
    ).pack(pady=(18, 14))

    bf = tk.Frame(topo)
    bf.pack()
    tk.Button(bf, text="Cancelar", width=12, command=lambda: definir("cancelar")).pack(
        side="left", padx=6
    )
    tk.Button(bf, text="Substituir", width=12, command=lambda: definir("substituir")).pack(
        side="left", padx=6
    )
    tk.Button(bf, text="Concatenar", width=12, command=lambda: definir("concatenar")).pack(
        side="left", padx=6
    )

    janela_pai.wait_window(topo)
    return escolha[0]


def importar_banco_interativo(janela_pai):
    caminho = filedialog.askopenfilename(
        title="Selecionar banco de dados",
        filetypes=[
            ("Banco ParakeetEye", "*.zip *.pkl"),
            ("ZIP", "*.zip"),
            ("Pickle", "*.pkl"),
            ("Todos os arquivos", "*.*"),
        ],
    )
    if not caminho:
        return

    try:
        df_prev = extrair_dataframe_importacao(caminho)
    except Exception as e:
        messagebox.showerror("Importação", f"Não foi possível ler o arquivo:\n{e}")
        return

    if df_prev.empty:
        messagebox.showinfo(
            "Importação",
            "O arquivo selecionado não contém registros para importar.",
        )
        return

    try:
        if banco_possui_registros():
            modo = _pergunta_substituir_concatenar(janela_pai)
            if modo is None or modo == "cancelar":
                return
            if modo == "substituir":
                substituir_banco_de_arquivo(caminho)
                messagebox.showinfo(
                    "Importação",
                    "Banco de dados substituído com sucesso.",
                )
            else:
                concatenar_banco_de_arquivo(caminho)
                messagebox.showinfo(
                    "Importação",
                    "Dados concatenados ao banco atual com sucesso.",
                )
        else:
            substituir_banco_de_arquivo(caminho)
            messagebox.showinfo(
                "Importação",
                "Banco de dados carregado com sucesso.",
            )
    except Exception as e:
        messagebox.showerror("Importação", f"Falha na importação:\n{e}")