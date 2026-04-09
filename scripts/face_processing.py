import cv2
import numpy as np
from deepface import DeepFace

from constants import *

#------------------------------------------
# Geometria e utilidades de caixa
#------------------------------------------
def _centro_bbox(loc):
    top, right, bottom, left = loc
    return float((left + right) / 2.0), float((top + bottom) / 2.0)


def expandir_bbox(top, right, bottom, left, altura_img, largura_img, pad_frac=None):
    """Aumenta a caixa para não cortar queixo/orelha em poses de perfil."""
    pad = pad_frac if pad_frac is not None else EMBEDDING_BBOX_PAD
    h = max(1, bottom - top)
    w = max(1, right - left)
    pad_y = int(h * pad)
    pad_x = int(w * pad)
    top = max(0, int(top) - pad_y)
    left = max(0, int(left) - pad_x)
    bottom = min(altura_img, int(bottom) + pad_y)
    right = min(largura_img, int(right) + pad_x)
    return top, right, bottom, left


def _bbox_iou(loc_a, loc_b):
    ta, ra, ba, la = loc_a
    tb, rb, bb, lb = loc_b

    inter_left = max(la, lb)
    inter_top = max(ta, tb)
    inter_right = min(ra, rb)
    inter_bottom = min(ba, bb)

    if inter_right <= inter_left or inter_bottom <= inter_top:
        return 0.0

    inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
    area_a = max(1, (ra - la) * (ba - ta))
    area_b = max(1, (rb - lb) * (bb - tb))
    return float(inter_area / float(area_a + area_b - inter_area + 1e-8))

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
# Detecção de faces com DeepFace
#------------------------------------------
def detectar_faces(imagem_bgr, anti_spoofing=False, detector_backend=None):
    backend = detector_backend or DEEPFACE_DETECTOR_LEVE
    try:
        resultados = DeepFace.extract_faces(
            img_path=imagem_bgr,
            detector_backend=backend,
            enforce_detection=False,
            align=True,
            anti_spoofing=anti_spoofing,
        )
    except Exception:
        return []

    faces = []
    for face in resultados:
        area = face.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))
        if w <= 0 or h <= 0:
            continue

        top = y
        right = x + w
        bottom = y + h
        left = x

        faces.append(
            {
                "loc": (top, right, bottom, left),  # formato compatível com sistema atual
                "is_real": bool(face.get("is_real", True)),
                "antispoof_score": float(face.get("antispoof_score", 1.0)),
            }
        )
    return faces

#------------------------------------------
# Embedding com DeepFace
#------------------------------------------
def gerar_embedding(imagem_bgr, localizacao):
    top, right, bottom, left = localizacao
    h, w = imagem_bgr.shape[:2]
    top, right, bottom, left = expandir_bbox(top, right, bottom, left, h, w)
    top = max(0, int(top)); left = max(0, int(left))
    bottom = min(h, int(bottom)); right = min(w, int(right))
    if bottom <= top or right <= left:
        return None

    recorte_bgr = imagem_bgr[top:bottom, left:right]
    if recorte_bgr.size == 0:
        return None

    try:
        rep = DeepFace.represent(
            img_path=recorte_bgr,
            model_name=DEEPFACE_MODEL,
            detector_backend="skip",
            enforce_detection=False,
            normalization="ArcFace",
        )
    except Exception:
        return None

    if not rep:
        return None
    embedding = rep[0].get("embedding")
    if embedding is None:
        return None
    return np.array(embedding, dtype=np.float32)

#------------------------------------------
# Distância entre embeddings (cosseno)
#------------------------------------------
def distancia_embeddings(embeddings_conhecidos, embedding_alvo):
    if len(embeddings_conhecidos) == 0 or embedding_alvo is None:
        return np.array([], dtype=np.float32)

    alvo = np.asarray(embedding_alvo, dtype=np.float32)
    alvo_norm = np.linalg.norm(alvo) + 1e-8

    distancias = []
    for emb in embeddings_conhecidos:
        atual = np.asarray(emb, dtype=np.float32)
        denom = (np.linalg.norm(atual) * alvo_norm) + 1e-8
        cos_sim = float(np.dot(atual, alvo) / denom)
        distancias.append(1.0 - cos_sim)
    return np.asarray(distancias, dtype=np.float32)

#------------------------------------------
# Associação de tracks por IoU
#------------------------------------------
def associar_tracks_por_iou(tracks_anteriores, localizacoes_atuais, iou_min=WEBCAM_IOU_MATCH):
    if not tracks_anteriores:
        return {}, set()

    matched = {}
    usados = set()
    for idx, loc in enumerate(localizacoes_atuais):
        melhor_id = None
        melhor_iou = 0.0
        for track_id, data in tracks_anteriores.items():
            if track_id in usados:
                continue
            iou = _bbox_iou(loc, data["loc"])
            if iou > melhor_iou:
                melhor_iou = iou
                melhor_id = track_id
        if melhor_id is not None and melhor_iou >= iou_min:
            matched[idx] = melhor_id
            usados.add(melhor_id)
    return matched, usados


def associar_tracks_robusto(tracks_anteriores, localizacoes_atuais, iou_min=None, max_dist_centro=None):
    """
    IoU sozinho perde o track quando a face gira (caixa muda de forma).
    Complementa com associação por proximidade do centro do rosto.
    """
    iou_min = iou_min if iou_min is not None else WEBCAM_IOU_MATCH
    max_dist_centro = max_dist_centro if max_dist_centro is not None else WEBCAM_TRACK_MAX_DIST

    matched, usados = associar_tracks_por_iou(tracks_anteriores, localizacoes_atuais, iou_min=iou_min)

    indices_sem_match = [i for i in range(len(localizacoes_atuais)) if i not in matched]
    tracks_livres = [tid for tid in tracks_anteriores.keys() if tid not in usados]

    for idx in indices_sem_match:
        loc = localizacoes_atuais[idx]
        cx, cy = _centro_bbox(loc)
        melhor_id = None
        melhor_d = 10**9
        for tid in tracks_livres:
            ox, oy = _centro_bbox(tracks_anteriores[tid]["loc"])
            d = float(np.hypot(cx - ox, cy - oy))
            if d < melhor_d:
                melhor_d = d
                melhor_id = tid
        if melhor_id is not None and melhor_d <= max_dist_centro:
            matched[idx] = melhor_id
            usados.add(melhor_id)
            tracks_livres.remove(melhor_id)

    return matched, usados

#------------------------------------------
# Verificação de profundidade / vivacidade
#------------------------------------------
def verificar_profundidade_face(
    imagem_rgb,
    localizacao,
    gray_anterior=None,
    exigir_movimento=False,
    imagem_bgr=None,
    anti_spoofing_info=None,
):
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

    # Anti-spoofing prioritário via DeepFace.
    if anti_spoofing_info is None:
        imagem_bgr = imagem_bgr if imagem_bgr is not None else cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)
        faces = detectar_faces(
            imagem_bgr,
            anti_spoofing=True,
            detector_backend=DEEPFACE_DETECTOR_PESADO,
        )
        melhor = None
        menor_delta = 10**9
        for face in faces:
            t, r, b, l = face["loc"]
            delta = abs(t - top) + abs(r - right) + abs(b - bottom) + abs(l - left)
            if delta < menor_delta:
                menor_delta = delta
                melhor = face
        anti_spoofing_info = melhor

    if anti_spoofing_info is None:
        if ANTISPOOF_MODO_ESTRITO:
            return False, "Não foi possível validar anti-spoofing da face."
        return True, "Anti-spoofing indisponível (modo tolerante)."

    is_real = bool(anti_spoofing_info.get("is_real", True))
    spoof_score = float(anti_spoofing_info.get("antispoof_score", 1.0))
    if (not is_real) or spoof_score < LIMIAR_ANTISPOOF_SCORE:
        if ANTISPOOF_MODO_ESTRITO:
            return False, "DeepFace anti-spoofing identificou possível foto/tela."
        return True, "Anti-spoofing inconclusivo (modo tolerante)."

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