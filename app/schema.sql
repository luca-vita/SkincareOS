-- Schema DDL per SQLite (WAL Mode abilitato, Foreign Keys ON)
-- Rimosso ogni riferimento a Infomaniak/cloud storage.

CREATE TABLE IF NOT EXISTS daily_logs (
    entry_date TEXT PRIMARY KEY, -- Formato: YYYY-MM-DD
    am_cleanser INTEGER NOT NULL CHECK(am_cleanser IN (0, 1)),
    am_azid INTEGER NOT NULL CHECK(am_azid IN (0, 1)),
    am_rederma INTEGER NOT NULL CHECK(am_rederma IN (0, 1)),
    am_spf INTEGER NOT NULL CHECK(am_spf IN (0, 1)),
    pm_cleanser INTEGER NOT NULL CHECK(pm_cleanser IN (0, 1)),
    pm_differin INTEGER NOT NULL CHECK(pm_differin IN (0, 1)),
    pm_rederma INTEGER NOT NULL CHECK(pm_rederma IN (0, 1)),
    stinging_index INTEGER NOT NULL CHECK(stinging_index BETWEEN 0 AND 3),
    gym_workout TEXT NOT NULL CHECK(gym_workout IN ('NONE', 'MORNING', 'AFTERNOON', 'EVENING')),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS photo_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_date TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    resolution_width INTEGER NOT NULL DEFAULT 2048,
    resolution_height INTEGER NOT NULL DEFAULT 2048,
    interpupillary_distance REAL NOT NULL,
    pih_path TEXT,
    texture_path TEXT,
    analysis_json TEXT,
    analysis_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entry_date) REFERENCES daily_logs(entry_date) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_logs_date ON daily_logs(entry_date);
CREATE INDEX IF NOT EXISTS idx_photos_date ON photo_checkpoints(entry_date);
