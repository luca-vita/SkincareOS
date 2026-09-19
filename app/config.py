import os

# Dimensioni target per l'allineamento biometrico
CANVAS_SIZE = 2048
D_TARGET = 614.4  # 30% di 2048
T_MID_X = 1024.0
T_MID_Y = 790.0   # ~38.6% di 2048 (occhi un po' più in basso: più fronte, meno vuoto sotto il mento)

# Limite di inclinazione della testa (roll angle) in gradi
MAX_TILT_DEG = 30.0

# Path dei dati
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
DB_PATH = os.path.join(DATA_DIR, "app.db")
MODEL_PATH = os.path.join(DATA_DIR, "models", "face_landmarker.task")

# Assicuriamoci che le cartelle esistano
os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
