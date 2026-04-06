import cv2
import numpy as np
import face_recognition

from constants import *

#------------------------------------------
# Recorte seguro do rosto
#------------------------------------------
def _recorte_rosto_seguro(imagem_rgb, top, right, bottom, left):
    h, w = imagem_rgb.shape[:2]
    top = max(0, int(top))
    left = max(0, int(left))
    bottom = min(h, int(bottom))
    right = min(w, int(right))
    if bottom <= top or right <= left:
        return None
    return imagem_rgb[top:bottom, left:right]

#------------------------------------------
# Variância do Laplaciano
#------------------------------------------
def variancia_laplaciano_rosto(imagem_rgb, localizacao):
    top, right, bottom, left = localizacao
    recorte = _recorte_rosto_seguro(imagem_rgb, top, right, bottom, left)
    if recorte is None or recorte.size == 0:
        return 0.0
    if len(recorte.shape) == 3:
        cinza = cv2.cvtColor(recorte, cv2.COLOR_RGB2GRAY)
    else:
        cinza = recorte
    return float(cv2.Laplacian(cinza, cv2.CV_64F).var())

#------------------------------------------
# Proporção do rosto na imagem
#------------------------------------------
def proporcao_rosto_na_imagem(imagem_rgb, localizacao):
    altura, largura = imagem_rgb.shape[:2]
    top, right, bottom, left = localizacao
    area_rosto = max(0, bottom - top) * max(0, right - left)
    area_img = altura * largura
    if area_img <= 0:
        return 0.0
    return area_rosto / float(area_img)

#------------------------------------------
# Movimento médio no ROI (Região de Interesse) do rosto
#------------------------------------------
def movimento_medio_roi(gray_atual, gray_anterior, top, right, bottom, left):
    if gray_anterior is None or gray_atual is None:
        return None
    ha, wa = gray_atual.shape[:2]
    hb, wb = gray_anterior.shape[:2]
    if ha != hb or wa != wb:
        return None
    top = max(0, int(top))
    left = max(0, int(left))
    bottom = min(ha, int(bottom))
    right = min(wa, int(right))
    if bottom <= top or right <= left:
        return None
    a = gray_atual[top:bottom, left:right].astype(np.float32)
    b = gray_anterior[top:bottom, left:right].astype(np.float32)
    if a.shape != b.shape or a.size == 0:
        return None
    return float(np.mean(np.abs(a - b)))

#------------------------------------------
# Verificação de profundidade / vivacidade
#------------------------------------------
def verificar_profundidade_face(imagem_rgb, localizacao, gray_anterior=None, exigir_movimento=False):
    """
    Função anti-foto/plano: variância do Laplaciano no rosto permite detectar fotos/planos.
    Se exigir_movimento, movimento médio no ROI do rosto permite detectar movimento.
    Retorna (ok, mensagem): 
    
    ok=True se a face é válida
    ok=False se a face é inválida

    mensagem: mensagem de erro se a face é inválida
    """
    top, right, bottom, left = localizacao
    prop = proporcao_rosto_na_imagem(imagem_rgb, localizacao)
    if prop < PROPORCAO_ROSTO_MIN or prop > PROPORCAO_ROSTO_MAX:
        return False, "Rosto com tamanho atípico (possível foto ou enquadramento inválido)."

    var_lap = variancia_laplaciano_rosto(imagem_rgb, localizacao)
    altura_rosto = max(1, bottom - top)
    limite_adaptativo = LIMIAR_LAPLACIANO_MIN * (100.0 / float(altura_rosto)) ** 0.35
    limite_adaptativo = max(LIMIAR_LAPLACIANO_MIN, min(limite_adaptativo, 120.0))

    if var_lap < limite_adaptativo:
        return False, "Baixa textura ou foco (possível impressao, tela ou plano sem profundidade aparente)."

    if exigir_movimento and gray_anterior is not None:
        cinza_atual = cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2GRAY) if len(imagem_rgb.shape) == 3 else imagem_rgb
        mov = movimento_medio_roi(cinza_atual, gray_anterior, top, right, bottom, left)
        if mov is not None and mov < LIMIAR_MOVIMENTO_ROI:
            return False, "Sem movimento detectado no rosto (possível foto estática)."

    return True, ""

#------------------------------------------
# Permitir apenas um rosto para cadastro
#------------------------------------------
def validar_unico_rosto_para_cadastro(localizacoes):
    """Cadastro: exige exatamente um rosto."""
    n = len(localizacoes)
    if n == 0:
        return False, "Nenhum rosto detectado."
    if n > 1:
        return False, "Várias faces detectadas. Cadastre apenas uma pessoa por vez."
    return True, ""