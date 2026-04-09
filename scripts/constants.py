# -------------------------------
# Thresholds de vivacidade / validação
# -------------------------------
LIMIAR_MOVIMENTO_ROI = 2.8
PROPORCAO_ROSTO_MIN = 0.12
PROPORCAO_ROSTO_MAX = 0.95
LIMIAR_ANTISPOOF_SCORE = 0.10
ANTISPOOF_MODO_ESTRITO = False
USAR_ANTISPOOF_CADASTRO = False

# -------------------------------
# Thresholds de reconhecimento (distância cosseno)
# ArcFace costuma operar bem entre ~0.30 e 0.45
# -------------------------------
LIMIAR_RECONHECIMENTO = 0.40
# Webcam: perfis / rotação pioram o embedding; limiar um pouco mais tolerante
LIMIAR_RECONHECIMENTO_WEBCAM = 0.43
LIMIAR_DIVERSIDADE_CADASTRO = 0.20

# -------------------------------
# Configuração DeepFace
# -------------------------------
DEEPFACE_MODEL = "ArcFace"
DEEPFACE_DETECTOR_PESADO = "retinaface"
# opencv: mais rápido; mtcnn: bom equilíbrio FPS x perfil; retinaface: máxima qualidade, mais pesado
DEEPFACE_DETECTOR_LEVE = "opencv"
# MTCNN: mais leve que RetinaFace na webcam/cadastro, ainda aceita rosto de lado razoavelmente bem
DEEPFACE_DETECTOR_TEMPO_REAL = "mtcnn"

# -------------------------------
# Performance da webcam
# -------------------------------
# Escala: maior ajuda perfil, menor = mais FPS (0.44–0.50 é um bom meio-termo)
WEBCAM_SCALE = 0.46
# Rodar detecção DeepFace só a cada N frames (principal ganho de FPS)
WEBCAM_DETECT_EVERY = 5
# Embedding ArcFace é caro: espaçar bastante se a máquina for fraca
WEBCAM_EMBED_EVERY = 16
# Anti-spoofing pesado: manter raro
WEBCAM_ANTISPOOF_EVERY = 90
WEBCAM_IOU_MATCH = 0.25
# Quando a face gira, IoU cai; usa-se também proximidade do centro (pixels no frame reduzido)
WEBCAM_TRACK_MAX_DIST = 72
# Margem no recorte antes do embedding (orelhábria / rosto parcial)
EMBEDDING_BBOX_PAD = 0.18
# Cadastro: rodar detector pesado só a cada N frames (ganho de FPS)
CADASTRO_DETECT_EVERY = 2