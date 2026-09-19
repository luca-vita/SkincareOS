# SkinCare OS - Clinical Skincare Engine & Biometric Timeline Tracker

SkinCare OS è un motore clinico self-hosted e privacy-preserving progettato per registrare trattamenti farmacologici per l'acne e tracciare la risposta dei tessuti nel tempo. La piattaforma elabora checkpoint fotografici ad alta risoluzione ($2048 \times 2048$ px), standardizza l'allineamento facciale tramite normalizzazione affine biometrica (basata su MediaPipe Face Mesh) e renderizza uno scrubber temporale discreto a latenza zero.

Tutti i dati e le foto rimangono esclusivamente in locale sul filesystem e su database SQLite.

---

## 🚀 Guida di Configurazione e Avvio

### 1. Prerequisiti
Assicurati di avere installato [Conda](https://docs.conda.io/en/latest/) (o Miniconda/Anaconda) sul tuo sistema.

### 2. Creazione dell'Ambiente Virtuale (Conda)
Poiché MediaPipe non supporta ufficialmente Python 3.13 a causa di modifiche ABI di CPython, è richiesto l'uso di **Python 3.11** o **3.12**.

Esegui i seguenti comandi nel tuo terminale:

```bash
# Crea un nuovo ambiente conda con Python 3.11
conda create -n skincare python=3.11 -y

# Attiva l'ambiente appena creato
conda activate skincare

# Installa le dipendenze richieste
pip install -r requirements.txt
```

### 3. Avvio del Server FastAPI
Avvia il server di sviluppo tramite Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Il server sarà accessibile all'indirizzo:
- Locale: `http://localhost:8000`
- Nella tua LAN: `http://<IP_DEL_PC>:8000` (ideale per accedere dal browser del tuo smartphone)

### 4. Configurazione routine / prodotti (dati personali)

Prodotti, attivi EMA, bande di saturazione, lag e target di aderenza **non** sono nel codice.

```bash
mkdir -p data
cp config.example.json data/routine.json
# modifica data/routine.json con i tuoi prodotti / α / target
```

- `config.example.json` — template condivisibile (nel repo)
- `data/routine.json` — config personale (cartella `data/` gitignored)
- Path override: env `ROUTINE_CONFIG_PATH`

Gli `id` degli step in `routine.am` / `routine.pm` devono coincidere con le colonne booleane di `daily_logs` (es. `am_azid`, `pm_differin`).

Se `data/routine.json` manca, l’app usa `config.example.json` come fallback.

### 5. Database

Non serve creare il DB a mano. Al primo `uvicorn`, `init_db()`:
1. crea `data/` se non esiste
2. apre/crea `data/app.db`
3. applica `app/schema.sql` (`CREATE TABLE IF NOT EXISTS …`)

Uno clone fresco parte con DB vuoto ma schema già pronto. Log, foto e metriche si popolano dall’uso. Le foto allineate finiscono in `data/photos/` (anch’essa sotto `data/`, quindi non in git).

---

## 🧪 Esecuzione dei Test

Per eseguire la suite di test unitari e di integrazione (che coprono la matematica della normalizzazione affine, i quality gate e i contratti delle API):

```bash
# Assicurati che l'ambiente sia attivo
conda activate skincare

# Esegui i test con pytest
PYTHONPATH=. pytest -v
```

---

## 📸 Test End-to-End con Foto Reale

I test automatici utilizzano mock per non dipendere dal modello di machine learning di MediaPipe a runtime. Per testare la pipeline biometrica reale con una foto di un volto:

1. Avvia il server (`uvicorn app.main:app ...`).
2. Apri il browser all'indirizzo `http://localhost:8000`.
3. Compila un log per la data odierna.
4. Carica una foto reale del tuo viso (in primo piano, frontale, ben illuminata e con inclinazione della testa inferiore a 30°).
5. Se la foto rispetta i criteri di qualità, verrà allineata istantaneamente e inserita nella **Timeline Biometrica** in basso, dove potrai usare lo slider per vederla.

In alternativa, puoi inviare una richiesta tramite `curl` da terminale:

```bash
curl -X POST "http://localhost:8000/api/v1/photos/upload?entry_date=2026-09-08" \
  -F "file=@/path/to/your/selfie.jpg"
```

---

## 📁 Struttura del Progetto

- `app/`
  - `main.py`: Punto di ingresso dell'applicazione, configurazione CORS, gestione eccezioni centralizzata e montaggio file statici.
  - `config.py`: Costanti geometriche di allineamento (2048px, interpupillare 614.4px, Y occhi 790px), limiti di tolleranza ed impostazione dei percorsi.
  - `database.py`: Gestione delle connessioni SQLite (WAL mode, Foreign Keys ON, `check_same_thread=False` per concurrency) e query CRUD/Timeline.
  - `models.py`: Modelli di dati Pydantic per la validazione dei log e della timeline.
  - `exceptions.py`: Eccezioni custom per la gestione degli errori del Quality Gate e del database.
  - `services/`: Orchestrazione app (`photo_analysis.py` salva WebP + PIH/texture; `clinical_analytics.py` dashboard EMA).
  - `schema.sql`: DDL del database SQLite.
  - `routers/`: Router FastAPI suddivisi per dominio (`logs.py`, `photos.py`, `timeline.py`, `metrics.py`).
- `cv_tools/`: Computer vision (MediaPipe, allineamento, ROI, PIH spectral, texture) + CLI.
- `analytics/`: Motore clinico EMA e plot dashboard.
- `static/`
  - `index.html`: Interfaccia utente SPA a pagina singola.
  - `style.css`: Stili CSS moderni e responsive ottimizzati per mobile.
  - `app.js`: Logica client-side con gestione dello stato, prefetch asincrono delle immagini in memoria heap e scrubbing a zero latenza (60fps).
- `tests/`
  - `test_cv_pipeline.py`: Test unitari su affine / quality gate (`cv_tools.align_photo`).
  - `test_api.py`: Test di integrazione API tramite `TestClient`.
  - `test_photo_analysis.py` / `test_clinical_engine.py`: orchestrazione analisi e motore EMA.
- `data/`: Cartella locale in cui vengono memorizzati il database SQLite (`app.db`) e le foto WebP normalizzate (`photos/`).
