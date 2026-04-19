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
LIMIAR_RECONHECIMENTO_WEBCAM = 0.43
LIMIAR_DIVERSIDADE_CADASTRO = 0.20

# -------------------------------
# Configuração DeepFace
# -------------------------------
DEEPFACE_MODEL = "ArcFace"
DEEPFACE_DETECTOR_PESADO = "retinaface"
DEEPFACE_DETECTOR_LEVE = "opencv"
DEEPFACE_DETECTOR_TEMPO_REAL = "mtcnn"

# -------------------------------
# Performance da webcam
# -------------------------------
WEBCAM_SCALE = 0.46
WEBCAM_DETECT_EVERY = 5
WEBCAM_EMBED_EVERY = 16
WEBCAM_ANTISPOOF_EVERY = 90
WEBCAM_LIVENESS_EVERY = 8
WEBCAM_SPOOF_FAIL_TOLERANCE = 2
WEBCAM_IOU_MATCH = 0.25
WEBCAM_TRACK_MAX_DIST = 72
WEBCAM_MAX_EMBEDS_PER_FRAME = 1
WEBCAM_USAR_ANTISPOOF_PESADO = False
EMBEDDING_BBOX_PAD = 0.18
CADASTRO_DETECT_EVERY = 2
