import cv2

# -------------------------------
# imagem redimensionada
# -------------------------------
def mostrar_imagem_redimensionada(imagem, nome_janela="Imagem"):
    try:
        altura_max = 700
        largura_max = 1000

        h, w = imagem.shape[:2]
        escala = min(largura_max / w, altura_max / h)

        nova_largura = int(w * escala)
        nova_altura = int(h * escala)

        imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))

        cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
        cv2.imshow(nome_janela, imagem_redimensionada)
    except:
        pass