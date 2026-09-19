import os
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import DATA_DIR, PHOTOS_DIR, BASE_DIR
from app.database import init_db
from app.exceptions import SkincareEngineException, DatabaseBusyException
from app.routers import logs, photos, timeline, metrics

app = FastAPI(
    title="Skincare Engine & Biometric Timeline Tracker",
    description="Engine locale per il tracciamento e l'allineamento biometrico dei checkpoint fotografici.",
    version="1.0.0"
)

# Configurazione CORS per consentire l'accesso da dispositivi nella stessa LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gestione centralizzata delle eccezioni dell'applicazione
@app.exception_handler(SkincareEngineException)
async def skincare_exception_handler(request: Request, exc: SkincareEngineException):
    """
    Cattura le eccezioni custom dell'applicazione e le mappa nei codici di errore HTTP appropriati.
    - FaceCountException e PoseAngleException -> HTTP 422 Unprocessable Entity con payload {"detail": "CODE"}
    - InvalidImageException -> HTTP 400 Bad Request con payload {"detail": "CODE"}
    - DatabaseBusyException -> HTTP 503 Service Unavailable con payload {"detail": "CODE"}
    """
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    
    if exc.code == "INVALID_IMAGE_PAYLOAD":
        status_code = status.HTTP_400_BAD_REQUEST
    elif exc.code == "DATABASE_BUSY":
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.code, "message": exc.message}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    print("[ERROR] Unhandled exception occurred:")
    traceback.print_exc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "INTERNAL_SERVER_ERROR", "message": str(exc)}
    )

# Include i router delle API
app.include_router(logs.router)
app.include_router(photos.router)
app.include_router(timeline.router)
app.include_router(metrics.router)

# Inizializzazione del database all'avvio dell'applicazione
@app.on_event("startup")
def on_startup():
    init_db()

# Monta le foto salvate in modo che siano accessibili tramite URL statici (/static/photos/...)
# Nota: deve essere montato PRIMA di /static altrimenti /static intercetta la richiesta
app.mount("/static/photos", StaticFiles(directory=PHOTOS_DIR), name="photos")

# Monta i file statici del frontend
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Endpoint di favicon per evitare il 404 nei log
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    # Un cerchio blu con un "+" bianco al centro (stile clinico minimal)
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <circle cx="50" cy="50" r="45" fill="#3b82f6"/>
        <rect x="45" y="25" width="10" height="50" rx="5" fill="#ffffff"/>
        <rect x="25" y="45" width="50" height="10" rx="5" fill="#ffffff"/>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")

# Endpoint di fallback per servire l'index.html della SPA
@app.get("/")
def read_index():
    from fastapi.responses import FileResponse
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "Frontend SPA non trovato. Crea static/index.html"}
    )
