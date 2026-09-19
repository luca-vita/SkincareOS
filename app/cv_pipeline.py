import os
import math
import tempfile
import urllib.request
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Registra il supporto per i file HEIC/HEIF in Pillow
register_heif_opener()

from app.config import CANVAS_SIZE, D_TARGET, T_MID_X, T_MID_Y, MAX_TILT_DEG, PHOTOS_DIR, MODEL_PATH
from app.exceptions import FaceCountException, PoseAngleException, InvalidImageException

def download_model_if_missing():
    """Scarica il file del modello face_landmarker.task se non è presente localmente."""
    if not os.path.exists(MODEL_PATH):
        print(f"[+] Modello {MODEL_PATH} non trovato. Download in corso...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            # Scarica il file in una posizione temporanea e poi lo sposta per evitare download corrotti
            temp_model_path = MODEL_PATH + ".tmp"
            urllib.request.urlretrieve(url, temp_model_path)
            os.replace(temp_model_path, MODEL_PATH)
            print("[+] Download del modello completato con successo!")
        except Exception as e:
            if os.path.exists(MODEL_PATH + ".tmp"):
                os.remove(MODEL_PATH + ".tmp")
            raise RuntimeError(f"Impossibile scaricare il modello di MediaPipe da {url}: {str(e)}")

def calculate_affine_matrix(p_l: tuple[float, float], p_r: tuple[float, float]) -> tuple[np.ndarray, float, float]:
    """
    Funzione pura che calcola la matrice di trasformazione affine di similitudine (2x3).
    
    Parametri:
    - p_l: Coordinate (x, y) della pupilla sinistra nell'immagine (Landmark 473 - occhio destro del soggetto)
    - p_r: Coordinate (x, y) della pupilla destra nell'immagine (Landmark 468 - occhio sinistro del soggetto)
    
    Ritorna:
    - M: Matrice affine 2x3 (np.ndarray)
    - d_current: Distanza interpupillare corrente in pixel (float)
    - theta: Angolo di rotazione in radianti (float)
    """
    x_l, y_l = p_l
    x_r, y_r = p_r
    
    # Centro del segmento interpupillare (P_mid)
    x_mid = (x_l + x_r) / 2.0
    y_mid = (y_l + y_r) / 2.0
    
    # Vettore interpupillare
    dx = x_r - x_l
    dy = y_r - y_l
    
    # Angolo della linea degli occhi rispetto all'orizzontale
    theta = math.atan2(dy, dx)
    
    # Normalizza l'angolo nell'intervallo [-pi/2, pi/2] per gestire il mirroring (ambiguità di 180 gradi)
    # Se l'immagine è specchiata, dx sarà negativo e l'angolo sarà vicino a 180° o -180°.
    # Riportandolo nell'intervallo [-90°, 90°], evitiamo di ruotare la testa sottosopra.
    while theta > math.pi / 2:
        theta -= math.pi
    while theta < -math.pi / 2:
        theta += math.pi
    
    # Distanza interpupillare corrente
    d_current = math.hypot(dx, dy)
    if d_current == 0:
        raise ValueError("La distanza interpupillare corrente non può essere zero.")
        
    # Fattore di scala uniforme
    s = D_TARGET / d_current
    
    # Componenti della matrice di similitudine
    alpha = s * math.cos(theta)
    beta = s * math.sin(theta)
    
    # Calcolo dei termini di traslazione per mappare P_mid a T_mid (1024.0, T_MID_Y)
    tx = (1.0 - alpha) * x_mid - beta * y_mid + (T_MID_X - x_mid)
    ty = beta * x_mid + (1.0 - alpha) * y_mid + (T_MID_Y - y_mid)
    
    M = np.array([
        [alpha, beta, tx],
        [-beta, alpha, ty]
    ], dtype=np.float32)
    
    return M, d_current, theta

def load_and_normalize_image(image_bytes: bytes) -> np.ndarray:
    """
    Carica un'immagine dai byte in memoria, normalizza l'orientamento EXIF
    e la converte in un array RGB NumPy.
    """
    import io
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Normalizza l'orientamento EXIF (es. rotazioni da fotocamere mobile)
        img = ImageOps.exif_transpose(img)
        # Converte in RGB se necessario (es. RGBA o scala di grigi)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        raise InvalidImageException(f"Impossibile decodificare l'immagine: {str(e)}")

def run_biometric_pipeline(img_rgb: np.ndarray, face_mesh_detector=None) -> tuple[np.ndarray, float]:
    """
    Esegue l'intero pipeline biometrico:
    1. Rilevamento dei landmark facciali tramite MediaPipe Tasks (FaceLandmarker).
    2. Quality Gate (esattamente 1 volto, inclinazione testa <= 30°).
    3. Calcolo della matrice affine di similitudine.
    4. Warp dell'immagine a 2048x2048 tramite interpolazione Lanczos-4.
    
    Ritorna:
    - warped_img: Immagine normalizzata RGB 2048x2048 (np.ndarray)
    - d_current: Distanza interpupillare originale (float)
    """
    import cv2
    
    h, w, _ = img_rgb.shape
    owns_detector = face_mesh_detector is None
    
    try:
        # Se non viene passato un detector, ne creiamo uno temporaneo usando le nuove API Tasks
        if owns_detector:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Assicura che il modello sia presente localmente prima di caricarlo
            download_model_if_missing()
            
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=2 # Ne cerchiamo fino a 2 per il Quality Gate (se > 1 rigettiamo)
            )
            face_mesh_detector = vision.FaceLandmarker.create_from_options(options)
            
            # Converte l'immagine numpy in MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = face_mesh_detector.detect(mp_image)
        else:
            # Per i test unitari con il detector mockato
            results = face_mesh_detector.process(img_rgb)
        
        # Quality Gate: verifica il numero di volti rilevati
        if not results.face_landmarks:
            raise FaceCountException("NO_FACE_DETECTED", "Nessun volto rilevato nell'immagine.")
            
        num_faces = len(results.face_landmarks)
        if num_faces > 1:
            raise FaceCountException("MULTIPLE_FACES_DETECTED", f"Rilevati {num_faces} volti. È richiesto esattamente 1 volto.")
            
        landmarks = results.face_landmarks[0]
        
        # Estrae i landmark dell'iride
        # Con la nuova API Tasks, i landmark dell'iride sono sempre inclusi alla fine dei 468 landmark facciali (totale 478)
        # Landmark 468: Centro iride sinistra del soggetto (appare a destra nell'immagine -> P_R)
        # Landmark 473: Centro iride destra del soggetto (appare a sinistra nell'immagine -> P_L)
        lm_left_iris = landmarks[473]
        lm_right_iris = landmarks[468]
        
        # Converte le coordinate normalizzate (0.0 - 1.0) in pixel reali
        p_l = (lm_left_iris.x * w, lm_left_iris.y * h)
        p_r = (lm_right_iris.x * w, lm_right_iris.y * h)
        
        print(f"[DEBUG] Image dimensions: {w}x{h}")
        print(f"[DEBUG] Left Iris (473): {p_l}")
        print(f"[DEBUG] Right Iris (468): {p_r}")
        
        # Calcola la matrice affine e i parametri geometrici
        M, d_current, theta = calculate_affine_matrix(p_l, p_r)
        
        # Pose Condition: Verifica l'inclinazione della testa (roll angle)
        theta_deg = math.degrees(theta)
        print(f"[DEBUG] Calculated IPD (d_current): {d_current:.2f}px")
        print(f"[DEBUG] Calculated Theta (radians): {theta:.4f}, (degrees): {theta_deg:.2f}°")
        
        if abs(theta_deg) > MAX_TILT_DEG:
            raise PoseAngleException(
                f"Inclinazione della testa eccessiva ({abs(theta_deg):.1f}°). Il limite massimo è {MAX_TILT_DEG}°."
            )
            
        # Esegue la trasformazione affine con interpolazione Lanczos-4
        warped_img = cv2.warpAffine(
            img_rgb,
            M,
            (CANVAS_SIZE, CANVAS_SIZE),
            flags=cv2.INTER_LANCZOS4
        )
        
        return warped_img, d_current
    finally:
        # Chiude il detector SUBITO, mentre le API C di MediaPipe sono ancora vive.
        # Se lasciamo fare a FaceLandmarker.__del__ a fine processo, crasha con:
        # TypeError: 'NoneType' object is not callable
        if owns_detector and face_mesh_detector is not None:
            try:
                face_mesh_detector.close()
            except Exception:
                pass
            face_mesh_detector = None
            import gc
            gc.collect()


def save_image_atomically(img_rgb: np.ndarray, entry_date_str: str) -> str:
    """
    Salva l'immagine normalizzata in formato WebP (qualità 90, lossy)
    in modo atomico sul filesystem locale.
    
    Ritorna il path del file salvato.
    """
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    dest_path = os.path.join(PHOTOS_DIR, f"{entry_date_str}.webp")
    
    # Crea un file temporaneo nella stessa cartella per garantire l'atomicità dello spostamento
    with tempfile.NamedTemporaryFile(suffix=".webp", dir=PHOTOS_DIR, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
    try:
        # Converte l'array numpy RGB in PIL Image e salva
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(tmp_path, format="WEBP", quality=90, lossless=False)
        
        # Spostamento atomico (sovrascrive se esiste già, permettendo retake)
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e
