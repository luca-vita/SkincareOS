/**
 * SkinCare OS - Client-Side SPA State Engine
 * Gestione dello stato reattivo, prefetching delle immagini, scrubbing, calendario interattivo e grafici SVG.
 */

// 1. App State Store
const state = {
    manifest: [],               // Array di TimelineItem ordinati per data
    memoryCache: new Map(),     // Mappa cacheKey -> HTMLImageElement
    allLogs: [],                // Storico completo di tutti i log giornalieri
    overlayMode: 'off',         // 'off' | 'pih' | 'texture'
    currentIndex: -1,           // Indice del checkpoint correntemente visualizzato nella timeline
    currentCalendarDate: new Date(), // Mese correntemente visualizzato nel calendario
    resultsPages: [],           // [page1_b64, page2_b64] plot risultati
    resultsPageIndex: 0,
    routineConfig: null         // prodotti / attivi / target da /api/v1/metrics/routine-config
};

// 2. DOM Elements Cache
const elements = {
    // Form Log Giornaliero
    logForm: document.getElementById('log-form'),
    logFormTitle: document.getElementById('log-form-title'),
    logDate: document.getElementById('log-date'),
    amRoutineSteps: document.getElementById('am-routine-steps'),
    pmRoutineSteps: document.getElementById('pm-routine-steps'),
    
    stingingIndex: document.getElementById('stinging-index'),
    gymWorkout: document.getElementById('gym-workout'),
    logNotes: document.getElementById('log-notes'),
    logMessage: document.getElementById('log-message'),
    submitLogBtn: document.getElementById('submit-log-btn'),

    // Form Upload Foto
    uploadForm: document.getElementById('upload-form'),
    uploadDate: document.getElementById('upload-date'),
    photoFile: document.getElementById('photo-file'),
    fileInfoText: document.getElementById('file-info-text'),
    uploadMessage: document.getElementById('upload-message'),
    submitUploadBtn: document.getElementById('submit-upload-btn'),
    deletePhotoBtn: document.getElementById('delete-photo-btn'),

    // Timeline Viewport & Slider
    viewportPlaceholder: document.getElementById('viewport-placeholder'),
    timelineImage: document.getElementById('timeline-image'),
    timelineImageBack: document.getElementById('timeline-image-back'),
    crossfadeToggle: document.getElementById('crossfade-toggle'),
    overlayModeButtons: document.querySelectorAll('.overlay-mode-btn'),
    viewportOverlay: document.getElementById('viewport-overlay'),
    timelineSlider: document.getElementById('timeline-slider'),
    sliderStartDate: document.getElementById('slider-start-date'),
    sliderCurrentInfo: document.getElementById('slider-current-info'),
    sliderEndDate: document.getElementById('slider-end-date'),

    checkpointAnalysisCard: document.getElementById('checkpoint-analysis-card'),
    checkpointAnalysisKicker: document.getElementById('checkpoint-analysis-kicker'),
    checkpointAnalysisTitle: document.getElementById('checkpoint-analysis-title'),
    checkpointAnalysisDate: document.getElementById('checkpoint-analysis-date'),
    checkpointAnalysisGrid: document.getElementById('checkpoint-analysis-grid'),

    // Navigazione Schede (Tabs)
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),

    // Calendario
    calendarTitle: document.getElementById('calendar-title'),
    calendarDaysGrid: document.getElementById('calendar-days-grid'),
    prevMonthBtn: document.getElementById('prev-month-btn'),
    nextMonthBtn: document.getElementById('next-month-btn'),

    // Dashboard clinica
    clinicalSatCards: document.getElementById('clinical-sat-cards'),
    clinicalWindowDays: document.getElementById('clinical-window-days'),
    clinicalResultsWindowDays: document.getElementById('clinical-results-window-days'),
    clinicalResultsImg: document.getElementById('clinical-results-img'),
    clinicalResultsEmpty: document.getElementById('clinical-results-empty'),
    resultsPagePrev: document.getElementById('results-page-prev'),
    resultsPageNext: document.getElementById('results-page-next'),
    resultsPageIndicator: document.getElementById('results-page-indicator'),
    clinicalCorrelationImg: document.getElementById('clinical-correlation-img'),
    clinicalCorrelationEmpty: document.getElementById('clinical-correlation-empty')
};

// Mappatura descrizioni e classi per Stinging Index

// Nomi dei mesi in italiano
const monthNames = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre"
];

// Inizializzazione date di default a oggi (YYYY-MM-DD)
const todayStr = new Date().toISOString().split('T')[0];
if (elements.logDate) elements.logDate.value = todayStr;
if (elements.uploadDate) elements.uploadDate.value = todayStr;

// 3. Gestione Inizializzazione e Stato

/**
 * URL immagine per un checkpoint in base alla modalità overlay attiva.
 */
function srcFor(item, mode = state.overlayMode) {
    if (mode === 'pih' && item.pih_url) return item.pih_url;
    if (mode === 'texture' && item.texture_url) return item.texture_url;
    return item.photo_url;
}

function cacheKeyFor(item, mode = state.overlayMode) {
    return `${item.entry_date}|${mode}`;
}

/** id DB `am_azid` → DOM id `am-azid` */
function slotDomId(slotId) {
    return String(slotId).replace(/_/g, '-');
}

function allRoutineSteps() {
    const r = state.routineConfig?.routine || {};
    return [...(r.am || []), ...(r.pm || [])];
}

function allSlotIds() {
    return allRoutineSteps().map((s) => s.id);
}

function getSlotCheckbox(slotId) {
    return document.getElementById(slotDomId(slotId));
}

function renderRoutineSteps() {
    const cfg = state.routineConfig;
    if (!cfg) return;
    [['am', elements.amRoutineSteps], ['pm', elements.pmRoutineSteps]].forEach(([period, container]) => {
        if (!container) return;
        const steps = cfg.routine?.[period] || [];
        container.innerHTML = steps.map((step) => {
            const domId = slotDomId(step.id);
            const desc = step.desc
                ? ` <span class="step-desc">(${step.desc})</span>`
                : '';
            return `
                <div class="form-group checkbox-group">
                    <label class="checkbox-label">
                        <input type="checkbox" id="${domId}" data-slot-id="${step.id}">
                        <span class="custom-checkbox"></span>
                        ${step.label || step.id}${desc}
                    </label>
                </div>`;
        }).join('');
    });
}

async function loadRoutineConfig() {
    const response = await fetch('/api/v1/metrics/routine-config');
    if (!response.ok) {
        throw new Error(`routine-config HTTP ${response.status}`);
    }
    state.routineConfig = await response.json();
    renderRoutineSteps();
}

function buildLogPayload() {
    const payload = {
        entry_date: elements.logDate.value,
        stinging_index: parseInt(elements.stingingIndex.value, 10),
        gym_workout: elements.gymWorkout.value,
        notes: elements.logNotes.value || null
    };
    allSlotIds().forEach((sid) => {
        const el = getSlotCheckbox(sid);
        payload[sid] = !!(el && el.checked);
    });
    return payload;
}

function fillLogSlotsFrom(log) {
    allSlotIds().forEach((sid) => {
        const el = getSlotCheckbox(sid);
        if (el) el.checked = !!log[sid];
    });
}

function countLogCompliance(log) {
    const ids = allSlotIds();
    if (!ids.length) return { checked: 0, total: 0 };
    const checked = ids.reduce((n, sid) => n + (log[sid] ? 1 : 0), 0);
    return { checked, total: ids.length };
}

/**
 * Pre-carica un URL in memoryCache sotto chiave data|mode.
 */
function preloadUrl(cacheKey, url) {
    return new Promise((resolve) => {
        if (!url || state.memoryCache.has(cacheKey)) {
            resolve();
            return;
        }
        const img = new Image();
        img.src = url;
        img.onload = () => {
            state.memoryCache.set(cacheKey, img);
            resolve();
        };
        img.onerror = () => {
            console.error(`Errore caricamento: ${url}`);
            resolve();
        };
    });
}

/**
 * Carica tutti i dati necessari dal server (timeline e storico log).
 */
async function loadAppData() {
    try {
        state.memoryCache.clear();

        const logsResponse = await fetch('/api/v1/logs');
        if (logsResponse.ok) {
            state.allLogs = await logsResponse.json();
        }

        const timelineResponse = await fetch('/api/v1/timeline');
        if (timelineResponse.ok) {
            const rawTimeline = await timelineResponse.json();
            const cacheBuster = Date.now();
            state.manifest = rawTimeline.map(item => ({
                ...item,
                photo_url: item.photo_url ? `${item.photo_url}?t=${cacheBuster}` : null,
                pih_url: item.pih_url ? `${item.pih_url}?t=${cacheBuster}` : null,
                texture_url: item.texture_url ? `${item.texture_url}?t=${cacheBuster}` : null,
            }));
        }

        const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
        refreshActiveTab(activeTab);

        if (elements.uploadDate) {
            updateDeletePhotoButtonVisibility(elements.uploadDate.value);
        }

        updateOverlayModeButtons();

        if (state.manifest.length > 0) {
            elements.sliderCurrentInfo.textContent = "Pre-caricamento immagini...";

            const loadPromises = state.manifest.flatMap(item => {
                const jobs = [preloadUrl(cacheKeyFor(item, 'off'), item.photo_url)];
                if (state.overlayMode === 'pih' && item.pih_url) {
                    jobs.push(preloadUrl(cacheKeyFor(item, 'pih'), item.pih_url));
                }
                if (state.overlayMode === 'texture' && item.texture_url) {
                    jobs.push(preloadUrl(cacheKeyFor(item, 'texture'), item.texture_url));
                }
                return jobs;
            });

            await Promise.all(loadPromises);

            elements.timelineSlider.min = 0;
            elements.timelineSlider.max = state.manifest.length - 1;
            elements.timelineSlider.step = 1;
            elements.timelineSlider.disabled = false;

            elements.sliderStartDate.textContent = formatDate(state.manifest[0].entry_date);
            elements.sliderEndDate.textContent = formatDate(state.manifest[state.manifest.length - 1].entry_date);

            if (state.currentIndex === -1) {
                renderFrame(state.manifest.length - 1);
            } else {
                renderFrame(Math.min(state.currentIndex, state.manifest.length - 1));
            }
        } else {
            showEmptyTimeline();
        }

    } catch (error) {
        console.error("Errore durante il caricamento dei dati:", error);
    }
}

function updateOverlayModeButtons() {
    if (!elements.overlayModeButtons) return;
    const hasPih = state.manifest.some(i => i.pih_url);
    const hasTex = state.manifest.some(i => i.texture_url);
    if (state.overlayMode === 'pih' && !hasPih) state.overlayMode = 'off';
    if (state.overlayMode === 'texture' && !hasTex) state.overlayMode = 'off';
    elements.overlayModeButtons.forEach(btn => {
        const mode = btn.dataset.mode;
        btn.classList.toggle('active', mode === state.overlayMode);
        if (mode === 'pih') btn.disabled = !hasPih;
        else if (mode === 'texture') btn.disabled = !hasTex;
        else btn.disabled = false;
    });
}

async function setOverlayMode(mode) {
    if (mode === state.overlayMode) return;
    state.overlayMode = mode;
    updateOverlayModeButtons();

    if (state.manifest.length === 0) return;
    elements.sliderCurrentInfo.textContent = "Caricamento overlay...";
    await Promise.all(state.manifest.map(item => {
        const url = srcFor(item, mode);
        return preloadUrl(cacheKeyFor(item, mode), url);
    }));
    // Cambio mode: swap istantaneo, senza crossfade (quella è solo tra checkpoint/date)
    renderFrame(
        state.currentIndex >= 0 ? state.currentIndex : state.manifest.length - 1,
        { crossfade: false }
    );
}

function showEmptyTimeline() {
    state.manifest = [];
    state.currentIndex = -1;
    elements.timelineSlider.disabled = true;
    elements.timelineSlider.value = 0;
    elements.sliderStartDate.textContent = "-";
    elements.sliderEndDate.textContent = "-";
    elements.sliderCurrentInfo.textContent = "Nessuna foto presente";
    
    elements.viewportPlaceholder.style.display = 'flex';
    elements.timelineImage.style.display = 'none';
    elements.timelineImageBack.style.display = 'none';
    elements.viewportOverlay.style.display = 'none';
    updateCheckpointAnalysisPanel(null);
}

function formatMetricValue(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    return Number(value).toFixed(digits);
}

function renderAnalysisMetricRows(rows) {
    if (!elements.checkpointAnalysisGrid) return;
    elements.checkpointAnalysisGrid.innerHTML = rows.map(row => `
        <div class="clinical-metric">
            <span class="metric-label">${row.label}</span>
            <span class="metric-value">${row.value}</span>
        </div>
    `).join('');
}

/**
 * Pannello metriche sotto la foto: solo in mode PIH / Texture.
 */
function updateCheckpointAnalysisPanel(item) {
    const card = elements.checkpointAnalysisCard;
    if (!card) return;

    const mode = state.overlayMode;
    if (!item || mode === 'off') {
        card.style.display = 'none';
        return;
    }

    card.style.display = 'block';
    if (elements.checkpointAnalysisDate) {
        elements.checkpointAnalysisDate.textContent = formatDate(item.entry_date);
    }

    const a = item.analysis;
    if (!a) {
        if (elements.checkpointAnalysisKicker) {
            elements.checkpointAnalysisKicker.textContent = 'ANALISI';
        }
        if (elements.checkpointAnalysisTitle) {
            elements.checkpointAnalysisTitle.textContent =
                mode === 'pih' ? 'PIH Spectral' : 'Texture';
        }
        if (elements.checkpointAnalysisGrid) {
            elements.checkpointAnalysisGrid.innerHTML =
                '<p class="checkpoint-analysis-empty">Nessuna analisi per questo checkpoint. Ri-carica la foto per generarla.</p>';
        }
        return;
    }

    if (mode === 'pih') {
        if (elements.checkpointAnalysisKicker) {
            elements.checkpointAnalysisKicker.textContent = 'RIEPILOGO';
        }
        if (elements.checkpointAnalysisTitle) {
            elements.checkpointAnalysisTitle.textContent = 'PIH Spectral';
        }
        renderAnalysisMetricRows([
            { label: 'Active Spots', value: String(a.active_spots_count ?? 0) },
            { label: 'Active Area (/aff.)', value: `${formatMetricValue(a.active_area_pct, 2)}%` },
            { label: 'Affected Area (/ROI)', value: `${formatMetricValue(a.affected_area_pct, 2)}%` },
            { label: 'Residual Area (/ROI)', value: `${formatMetricValue(a.residual_area_pct, 2)}%` },
            { label: 'Discromia Score', value: formatMetricValue(a.discromia_score, 1) },
            { label: 'Peak Δ (p98)', value: formatMetricValue(a.peak_p98, 4) },
        ]);
        return;
    }

    if (mode === 'texture') {
        if (elements.checkpointAnalysisKicker) {
            elements.checkpointAnalysisKicker.textContent = 'RIEPILOGO';
        }
        if (elements.checkpointAnalysisTitle) {
            elements.checkpointAnalysisTitle.textContent = 'Texture';
        }
        renderAnalysisMetricRows([
            { label: 'Pore Prominence', value: `${formatMetricValue(a.pore_prominence_pct, 2)}%` },
            { label: 'Roughness Index', value: formatMetricValue(a.roughness_index, 2) },
            { label: 'Smoothness Score', value: `${formatMetricValue(a.smoothness_score, 1)}/100` },
        ]);
    }
}

/**
 * Renderizza un frame specifico della timeline direttamente dalla cache (zero latenza).
 * Crossfade solo se si cambia checkpoint (data), non al cambio overlay Off/PIH/Texture.
 */
function renderFrame(index, options = {}) {
    if (index < 0 || index >= state.manifest.length) return;

    const allowCrossfade = options.crossfade !== false;
    const prevIndex = state.currentIndex;
    const checkpointChanged = prevIndex !== index && prevIndex >= 0;
    
    state.currentIndex = index;
    elements.timelineSlider.value = index;
    
    const item = state.manifest[index];
    const mode = state.overlayMode;
    const key = cacheKeyFor(item, mode);
    const cachedImg = state.memoryCache.get(key);
    const newSrc = cachedImg ? cachedImg.src : srcFor(item, mode);
    
    const crossfadeEnabled = elements.crossfadeToggle ? elements.crossfadeToggle.checked : false;
    const oldSrc = elements.timelineImage.src;
    const isImageVisible = elements.timelineImage.style.display !== 'none';
    
    if (
        allowCrossfade &&
        crossfadeEnabled &&
        checkpointChanged &&
        oldSrc &&
        oldSrc !== newSrc &&
        isImageVisible
    ) {
        elements.timelineImageBack.src = oldSrc;
        elements.timelineImageBack.style.display = 'block';
        
        elements.timelineImage.style.transition = 'none';
        elements.timelineImage.style.opacity = '0';
        elements.timelineImage.src = newSrc;
        elements.timelineImage.style.display = 'block';
        
        void elements.timelineImage.offsetWidth;
        
        elements.timelineImage.style.transition = 'opacity 250ms ease-in-out';
        elements.timelineImage.style.opacity = '1';
    } else {
        elements.timelineImageBack.style.display = 'none';
        elements.timelineImage.style.transition = 'none';
        elements.timelineImage.style.opacity = '1';
        elements.timelineImage.src = newSrc;
        elements.timelineImage.style.display = 'block';
    }
    
    elements.viewportPlaceholder.style.display = 'none';

    elements.sliderCurrentInfo.textContent = `Checkpoint ${index + 1} di ${state.manifest.length}`;
    updateCheckpointAnalysisPanel(item);
}

/**
 * Recupera i dati di pressione terapeutica dal server e disegna il radar plot.
 */
async function updateTherapeuticPressure(dateStr) {
    try {
        const response = await fetch(`/api/v1/metrics/therapeutic-pressure?target_date=${dateStr}`);
        if (response.ok) {
            const data = await response.json();
            drawRadarPlot(data);
            renderRadarMicroCards(data);
        }
    } catch (error) {
        console.error("Errore nel recupero della pressione terapeutica:", error);
    }
}

/**
 * Disegna il grafico radar a 3 assi su Canvas HTML5.
 */
function drawRadarPlot(data) {
    const canvas = elements.radarCanvas;
    if (!canvas) return;

    const axes = Array.isArray(data.axes) ? data.axes : [];
    if (!axes.length) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const getCssVar = (name, fallback) =>
        getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
    const accentColor = getCssVar('--accent', '#7c3aed');
    const textColor = getCssVar('--text-color', '#1e1b4b');
    const textMuted = getCssVar('--text-muted', '#5c5685');

    canvas.width = 320 * dpr;
    canvas.height = 320 * dpr;
    canvas.style.width = '320px';
    canvas.style.height = '320px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const width = 320;
    const height = 320;
    const cx = width / 2;
    const cy = height / 2 + 10;
    const R = 85;
    ctx.clearRect(0, 0, width, height);

    const n = axes.length;
    const angles = axes.map((_, i) => -Math.PI / 2 + (i * 2 * Math.PI) / n);

    function getCoords(value, angle) {
        return {
            x: cx + R * value * Math.cos(angle),
            y: cy + R * value * Math.sin(angle)
        };
    }

    const levels = [0.25, 0.5, 0.75, 1.0];
    levels.forEach((level) => {
        ctx.beginPath();
        angles.forEach((angle, i) => {
            const p = getCoords(level, angle);
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
        ctx.strokeStyle = 'rgba(216, 210, 241, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
        if (level === 1.0) {
            ctx.fillStyle = textMuted;
            ctx.font = '9px Comfortaa, sans-serif';
            ctx.textAlign = 'right';
            ctx.fillText('100%', cx - 5, cy - R + 3);
        }
    });

    angles.forEach((angle) => {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        const p = getCoords(1.0, angle);
        ctx.lineTo(p.x, p.y);
        ctx.strokeStyle = 'rgba(216, 210, 241, 0.6)';
        ctx.stroke();
    });

    // Target polygon
    ctx.beginPath();
    axes.forEach((axis, i) => {
        const p = getCoords((axis.target_pct || 0) / 100.0, angles[i]);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
    });
    ctx.closePath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);

    // Current polygon
    const verts = axes.map((axis, i) =>
        getCoords((axis.current_pct || 0) / 100.0, angles[i])
    );
    ctx.beginPath();
    verts.forEach((v, i) => {
        if (i === 0) ctx.moveTo(v.x, v.y);
        else ctx.lineTo(v.x, v.y);
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(124, 58, 237, 0.25)';
    ctx.fill();
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    verts.forEach((v) => {
        ctx.beginPath();
        ctx.arc(v.x, v.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = accentColor;
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    });

    axes.forEach((axis, i) => {
        const angle = angles[i];
        const edge = getCoords(1.0, angle);
        const label = axis.label || axis.id;
        const slotsUnit = axis.total_slots >= 20 ? 'sl' : 'gg';
        ctx.fillStyle = textColor;
        ctx.font = 'bold 10px Comfortaa, sans-serif';
        if (Math.abs(angle + Math.PI / 2) < 0.01) {
            ctx.textAlign = 'center';
            ctx.fillText(label, cx, cy - R - 20);
            ctx.font = '9px Comfortaa, sans-serif';
            ctx.fillStyle = textMuted;
            ctx.fillText(
                `${axis.current_pct}% (${axis.completed_slots}/${axis.total_slots} ${slotsUnit})`,
                cx,
                cy - R - 8
            );
        } else if (Math.cos(angle) >= 0) {
            ctx.textAlign = 'left';
            ctx.fillText(label, edge.x + 8, edge.y + 5);
            ctx.font = '9px Comfortaa, sans-serif';
            ctx.fillStyle = textMuted;
            ctx.fillText(
                `${axis.current_pct}% (${axis.completed_slots}/${axis.total_slots} ${slotsUnit})`,
                edge.x + 8,
                edge.y + 17
            );
        } else {
            ctx.textAlign = 'right';
            ctx.fillText(label, edge.x - 8, edge.y + 5);
            ctx.font = '9px Comfortaa, sans-serif';
            ctx.fillStyle = textMuted;
            ctx.fillText(
                `${axis.current_pct}% (${axis.completed_slots}/${axis.total_slots} ${slotsUnit})`,
                edge.x - 8,
                edge.y + 17
            );
        }
    });
}

function renderRadarMicroCards(data) {
    const container = elements.radarMicroCards;
    if (!container) return;
    container.innerHTML = '';
    const axes = Array.isArray(data.axes) ? data.axes : [];
    axes.forEach((axis) => {
        const card = document.createElement('div');
        const isCompliant = !!axis.is_compliant;
        card.className = `radar-micro-card ${isCompliant ? 'compliant' : 'non-compliant'}`;
        const slotsUnit = axis.total_slots >= 20 ? 'sl' : 'gg';
        card.innerHTML = `
            <span class="active-name">${axis.label || axis.id}</span>
            <span class="metric-val ${isCompliant ? 'compliant' : 'non-compliant'}">${axis.current_pct}%</span>
            <span class="target-val">Target: &ge;${axis.target_pct}% (${axis.completed_slots}/${axis.total_slots} ${slotsUnit})</span>
        `;
        container.appendChild(card);
    });
}

function refreshActiveTab(tabId) {
    if (tabId === 'calendar-tab') {
        renderCalendar();
    } else if (tabId === 'charts-tab') {
        renderChartsAndStats();
    }
}

// 4. Logica del Calendario Storico

/**
 * Renderizza la griglia del calendario mensile.
 */
function renderCalendar() {
    const year = state.currentCalendarDate.getFullYear();
    const month = state.currentCalendarDate.getMonth();

    // Imposta il titolo del mese
    elements.calendarTitle.textContent = `${monthNames[month]} ${year}`;

    // Pulisce la griglia precedente
    elements.calendarDaysGrid.innerHTML = '';

    // Primo giorno del mese e numero totale di giorni
    const firstDayIndex = new Date(year, month, 1).getDay(); // 0 = Dom, 1 = Lun...
    const totalDays = new Date(year, month + 1, 0).getDate();

    // Converte l'indice del primo giorno per far iniziare la settimana di Lunedì (0 = Lun, 6 = Dom)
    const startOffset = firstDayIndex === 0 ? 6 : firstDayIndex - 1;

    // Crea celle vuote per i giorni del mese precedente
    for (let i = 0; i < startOffset; i++) {
        const emptyCell = document.createElement('div');
        emptyCell.className = 'calendar-day empty';
        elements.calendarDaysGrid.appendChild(emptyCell);
    }

    // Mappa i log per data per un accesso rapido
    const logsMap = new Map(state.allLogs.map(log => [log.entry_date, log]));

    // Crea le celle per ciascun giorno del mese
    for (let day = 1; day <= totalDays; day++) {
        const dayCell = document.createElement('div');
        dayCell.className = 'calendar-day';

        // Costruisce la stringa della data (YYYY-MM-DD) locale
        const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
        
        // Numero del giorno
        const dayNumSpan = document.createElement('span');
        dayNumSpan.className = 'day-number';
        dayNumSpan.textContent = day;
        dayCell.appendChild(dayNumSpan);

        // Verifica se è oggi
        if (dateStr === todayStr) {
            dayCell.classList.add('today');
        }

        // Cerca se esiste un log per questo giorno
        const log = logsMap.get(dateStr);
        if (log) {
            // Calcola la compliance dagli step definiti in routine.json
            const { checked, total } = countLogCompliance(log);
            
            // Assegna la classe di compliance
            if (total > 0 && checked === total) {
                dayCell.classList.add('compliance-full');
            } else if (checked > 0) {
                dayCell.classList.add('compliance-partial');
            } else {
                dayCell.classList.add('compliance-none');
            }

            // Aggiunge indicatore dello Stinging Index (pallino colorato)
            const stingDot = document.createElement('span');
            stingDot.className = `stinging-dot stinging-${log.stinging_index}`;
            dayCell.appendChild(stingDot);

            // Indicatori icone (Palestra, Foto)
            const indicatorsContainer = document.createElement('div');
            indicatorsContainer.className = 'day-indicators';

            if (log.gym_workout !== 'NONE') {
                const gymIcon = document.createElement('span');
                gymIcon.textContent = '🏋️';
                gymIcon.title = `Allenamento: ${log.gym_workout}`;
                indicatorsContainer.appendChild(gymIcon);
            }

            if (log.has_photo) {
                const photoIcon = document.createElement('span');
                photoIcon.textContent = '📷';
                photoIcon.title = "Checkpoint Fotografico presente";
                photoIcon.style.fontSize = '0.7rem'; /* Rende l'icona leggermente più piccolina e proporzionata */
                indicatorsContainer.appendChild(photoIcon);
            }

            dayCell.appendChild(indicatorsContainer);
        }

        // Evento click: carica il log nel form di sinistra per visualizzazione/modifica
        dayCell.addEventListener('click', () => {
            selectDateForEditing(dateStr);
        });

        elements.calendarDaysGrid.appendChild(dayCell);
    }
}

/**
 * Seleziona una data specifica, carica il log corrispondente e precompila il form di sinistra.
 */
async function selectDateForEditing(dateStr) {
    elements.logDate.value = dateStr;
    
    // Allinea anche la data del form di upload della foto per comodità
    if (elements.uploadDate) {
        elements.uploadDate.value = dateStr;
        updateDeletePhotoButtonVisibility(dateStr);
    }
    
    // Cambia il titolo del form per indicare la modalità modifica
    const formatted = formatDate(dateStr);
    elements.logFormTitle.innerHTML = `Modifica Log <span class="accent">${formatted}</span>`;
    
    // Resetta i messaggi precedenti
    elements.logMessage.style.display = 'none';

    try {
        const response = await fetch(`/api/v1/logs/${dateStr}`);
        if (response.ok) {
            const log = await response.json();
            fillLogSlotsFrom(log);
            
            elements.stingingIndex.value = log.stinging_index;
            elements.gymWorkout.value = log.gym_workout;
            elements.logNotes.value = log.notes || '';
        } else {
            // Se non esiste, resetta il form per un nuovo inserimento su quella data
            resetLogFormFields();
        }
    } catch (error) {
        console.error("Errore caricamento log giornaliero:", error);
    }

    // Se non ci sono foto, aggiorna comunque il radar plot della data selezionata se siamo nella scheda grafici!
    const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
    if (activeTab === 'charts-tab') {
        updateTherapeuticPressure(dateStr);
    }

    // Scorri la pagina verso il form su dispositivi mobili
    if (window.innerWidth < 900) {
        elements.logForm.scrollIntoView({ behavior: 'smooth' });
    }
}

function resetLogFormFields() {
    allSlotIds().forEach((sid) => {
        const el = getSlotCheckbox(sid);
        if (el) el.checked = false;
    });
    elements.stingingIndex.value = "0";
    elements.gymWorkout.value = "NONE";
    elements.logNotes.value = '';
}

// 5. Logica dei Grafici SVG e Statistiche (Zero-Dependency)

/**
 * Genera e renderizza i grafici SVG e calcola le statistiche di correlazione.
 */
/**
 * Carica e mostra la dashboard clinica (Abitudini + Risultati)
 * per la data del form log (calendario / date picker).
 */
async function renderChartsAndStats() {
    const emptyRes = elements.clinicalResultsEmpty;
    const emptyCorr = elements.clinicalCorrelationEmpty;
    const imgRes = elements.clinicalResultsImg;
    const imgCorr = elements.clinicalCorrelationImg;
    const asOf = elements.logDate?.value || todayStr;
    let windowDays = parseInt(elements.clinicalWindowDays?.value || '30', 10);
    if (!Number.isFinite(windowDays)) windowDays = 30;
    windowDays = Math.min(90, Math.max(3, windowDays));
    if (elements.clinicalWindowDays) elements.clinicalWindowDays.value = String(windowDays);

    let resultsWindowDays = parseInt(elements.clinicalResultsWindowDays?.value || '20', 10);
    if (!Number.isFinite(resultsWindowDays)) resultsWindowDays = 20;
    resultsWindowDays = Math.min(90, Math.max(3, resultsWindowDays));
    if (elements.clinicalResultsWindowDays) {
        elements.clinicalResultsWindowDays.value = String(resultsWindowDays);
    }

    if (elements.clinicalSatCards) {
        elements.clinicalSatCards.innerHTML = '<p class="clinical-empty-msg">Generazione abitudini…</p>';
    }
    if (emptyRes) {
        emptyRes.style.display = 'block';
        emptyRes.textContent = 'Generazione sezione risultati…';
    }
    if (imgRes) imgRes.style.display = 'none';
    if (imgCorr) imgCorr.style.display = 'none';
    if (emptyCorr) emptyCorr.style.display = 'none';

    try {
        const response = await fetch(
            `/api/v1/metrics/clinical-dashboard?target_date=${encodeURIComponent(asOf)}&window_days=${windowDays}&results_window_days=${resultsWindowDays}`
        );
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        renderClinicalSatCards(
            data.latest_saturation || {},
            asOf,
            data.swimlane_pngs_base64 || {}
        );

        if (imgRes && (data.results_page1_png_base64 || data.results_page2_png_base64)) {
            state.resultsPages = [
                data.results_page1_png_base64 || null,
                data.results_page2_png_base64 || null
            ].filter(Boolean);
            state.resultsPageIndex = 0;
            showResultsPage(0);
            if (emptyRes) emptyRes.style.display = 'none';
        } else if (emptyRes) {
            state.resultsPages = [];
            emptyRes.textContent = 'Nessun checkpoint con metriche nel periodo.';
            if (imgRes) imgRes.style.display = 'none';
            updateResultsPagerUi();
        }

        if (imgCorr && data.correlation_png_base64) {
            imgCorr.src = `data:image/png;base64,${data.correlation_png_base64}`;
            imgCorr.style.display = 'block';
            if (emptyCorr) emptyCorr.style.display = 'none';
        }
    } catch (error) {
        console.error('Errore dashboard clinica:', error);
        if (elements.clinicalSatCards) {
            elements.clinicalSatCards.innerHTML = '<p class="clinical-empty-msg">Impossibile generare la sezione abitudini.</p>';
        }
        if (emptyRes) {
            emptyRes.style.display = 'block';
            emptyRes.textContent = 'Impossibile generare la sezione risultati.';
        }
    }
}

function showResultsPage(index) {
    const pages = state.resultsPages || [];
    if (!pages.length) {
        updateResultsPagerUi();
        return;
    }
    const i = Math.max(0, Math.min(pages.length - 1, index));
    state.resultsPageIndex = i;
    const imgRes = elements.clinicalResultsImg;
    if (imgRes) {
        imgRes.src = `data:image/png;base64,${pages[i]}`;
        imgRes.style.display = 'block';
    }
    updateResultsPagerUi();
}

function updateResultsPagerUi() {
    const pages = state.resultsPages || [];
    const i = state.resultsPageIndex || 0;
    const n = pages.length;
    if (elements.resultsPageIndicator) {
        elements.resultsPageIndicator.textContent = n ? `${i + 1} / ${n}` : '—';
    }
    if (elements.resultsPagePrev) {
        elements.resultsPagePrev.disabled = n < 2 || i <= 0;
    }
    if (elements.resultsPageNext) {
        elements.resultsPageNext.disabled = n < 2 || i >= n - 1;
    }
}

function renderClinicalSatCards(latest, asOf, swimlanes = {}) {
    const container = elements.clinicalSatCards;
    if (!container) return;
    const agentsCfg = state.routineConfig?.actives || {};
    const agents = Object.keys(agentsCfg);
    const dateLabel = asOf ? formatDate(asOf) : '';
    if (!latest || !Object.keys(latest).length || !agents.length) {
        container.innerHTML = `<p class="clinical-empty-msg">Nessuna saturazione EMA al ${dateLabel || 'giorno selezionato'}.</p>`;
        return;
    }
    container.innerHTML = agents.map((key) => {
            const info = latest[key] || { saturation_pct: 0, band: 'Critica', color: '#ef4444' };
            const label = agentsCfg[key]?.label || key;
            const laneB64 = swimlanes[key];
            const laneHtml = laneB64
                ? `<div class="clinical-sat-swim"><img src="data:image/png;base64,${laneB64}" alt="Andamento ${label}"></div>`
                : '';
            return `
            <div class="clinical-sat-card" style="border-color:${info.color}">
                <span class="clinical-sat-name">${label}</span>
                <span class="clinical-sat-value" style="color:${info.color}">${info.saturation_pct}%</span>
                <span class="clinical-sat-band">${info.band}</span>
                ${laneHtml}
            </div>`;
        }).join('');
}

/**
 * Formatta una data ISO (YYYY-MM-DD) in un formato più leggibile (DD/MM/YYYY).
 */
function formatDate(isoString) {
    if (!isoString) return "-";
    const parts = isoString.split('-');
    if (parts.length !== 3) return isoString;
    return `${parts[2]}/${parts[1]}/${parts[0]}`;
}

// 6. Event Listeners e Navigazione

// Gestione dello switch delle schede (Tabs)
elements.tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const targetTab = btn.dataset.tab;
        
        // Rimuove classe active da tutti i bottoni e contenuti
        elements.tabButtons.forEach(b => b.classList.remove('active'));
        elements.tabContents.forEach(c => c.classList.remove('active'));
        
        // Aggiunge classe active al bottone e contenuto selezionati
        btn.classList.add('active');
        document.getElementById(targetTab).classList.add('active');

        // Aggiorna la scheda appena attivata
        refreshActiveTab(targetTab);
    });
});

// Navigazione Mesi Calendario
elements.prevMonthBtn.addEventListener('click', () => {
    state.currentCalendarDate.setMonth(state.currentCalendarDate.getMonth() - 1);
    renderCalendar();
});

elements.nextMonthBtn.addEventListener('click', () => {
    state.currentCalendarDate.setMonth(state.currentCalendarDate.getMonth() + 1);
    renderCalendar();
});

// Aggiornamento della timeline sullo scrubbing dello slider (evento 'input' per reattività a 60fps)
elements.timelineSlider.addEventListener('input', (e) => {
    renderFrame(parseInt(e.target.value));
});

// Toggle overlay Off / PIH / Texture
if (elements.overlayModeButtons) {
    elements.overlayModeButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            setOverlayMode(btn.dataset.mode);
        });
    });
}

// Data del form log = riferimento dashboard clinica
if (elements.logDate) {
    elements.logDate.addEventListener('change', () => {
        const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
        if (activeTab === 'charts-tab') {
            renderChartsAndStats();
        }
    });
}

if (elements.clinicalWindowDays) {
    elements.clinicalWindowDays.addEventListener('change', () => {
        const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
        if (activeTab === 'charts-tab') {
            renderChartsAndStats();
        }
    });
}

if (elements.clinicalResultsWindowDays) {
    elements.clinicalResultsWindowDays.addEventListener('change', () => {
        const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
        if (activeTab === 'charts-tab') {
            renderChartsAndStats();
        }
    });
}

if (elements.resultsPagePrev) {
    elements.resultsPagePrev.addEventListener('click', () => {
        showResultsPage((state.resultsPageIndex || 0) - 1);
    });
}
if (elements.resultsPageNext) {
    elements.resultsPageNext.addEventListener('click', () => {
        showResultsPage((state.resultsPageIndex || 0) + 1);
    });
}

// Gestione selezione file per mostrare il nome del file
elements.photoFile.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        elements.fileInfoText.textContent = e.target.files[0].name;
    } else {
        elements.fileInfoText.textContent = "Nessun file selezionato (JPEG, PNG, HEIC)";
    }
});

// Data del form log = sorgente del radar (anche se cambiata a mano, non solo dal calendario)
if (elements.logDate) {
    elements.logDate.addEventListener('change', (e) => {
        const dateStr = e.target.value;
        if (!dateStr) return;
        const activeTab = document.querySelector('.tab-btn.active')?.dataset.tab;
        if (activeTab === 'charts-tab') {
            updateTherapeuticPressure(dateStr);
        }
    });
}

// Gestione cambio data nel form foto per mostrare/nascondere il pulsante Elimina
elements.uploadDate.addEventListener('change', (e) => {
    updateDeletePhotoButtonVisibility(e.target.value);
});

function updateDeletePhotoButtonVisibility(dateStr) {
    if (!elements.deletePhotoBtn) return;
    
    // Controlla se esiste una foto per questa data nei log attuali
    const hasPhoto = state.allLogs.some(log => log.entry_date === dateStr && log.has_photo);
    if (hasPhoto) {
        elements.deletePhotoBtn.style.display = 'block';
    } else {
        elements.deletePhotoBtn.style.display = 'none';
    }
}

// Gestione Eliminazione Foto
if (elements.deletePhotoBtn) {
    elements.deletePhotoBtn.addEventListener('click', async () => {
        const dateVal = elements.uploadDate.value;
        if (!dateVal) return;
        
        if (!confirm(`Sei sicuro di voler eliminare definitivamente il checkpoint fotografico del ${formatDate(dateVal)}?`)) {
            return;
        }
        
        showMessage(elements.uploadMessage, 'loading', 'Eliminazione foto in corso...');
        elements.deletePhotoBtn.disabled = true;
        elements.submitUploadBtn.disabled = true;
        
        try {
            const response = await fetch(`/api/v1/photos/delete?entry_date=${dateVal}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                showMessage(elements.uploadMessage, 'success', 'Foto eliminata con successo!');
                elements.photoFile.value = '';
                elements.fileInfoText.textContent = "Nessun file selezionato (JPEG, PNG, HEIC)";
                
                // Ricarica tutti i dati e aggiorna l'interfaccia
                await loadAppData();
                updateDeletePhotoButtonVisibility(dateVal);
            } else {
                const errData = await response.json();
                showMessage(elements.uploadMessage, 'error', `Errore: ${errData.detail || 'Impossibile eliminare la foto.'}`);
            }
        } catch (error) {
            showMessage(elements.uploadMessage, 'error', 'Errore di rete durante l\'eliminazione della foto.');
        } finally {
            elements.deletePhotoBtn.disabled = false;
            elements.submitUploadBtn.disabled = false;
        }
    });
}

// Invio del Log Giornaliero (Idempotente UPSERT)
elements.logForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const payload = buildLogPayload();

    showMessage(elements.logMessage, 'loading', 'Salvataggio in corso...');
    elements.submitLogBtn.disabled = true;

    try {
        const response = await fetch('/api/v1/logs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            showMessage(elements.logMessage, 'success', 'Log salvato con successo!');
            
            // Ripristina il titolo del form a "Registra Log Giornaliero"
            elements.logFormTitle.textContent = "Registra Log Giornaliero";
            
            // Ricarica tutti i dati dell'applicazione
            await loadAppData();
        } else {
            const errData = await response.json();
            showMessage(elements.logMessage, 'error', `Errore: ${errData.detail || 'Impossibile salvare il log.'}`);
        }
    } catch (error) {
        showMessage(elements.logMessage, 'error', 'Errore di rete durante il salvataggio del log.');
    } finally {
        elements.submitLogBtn.disabled = false;
    }
});

// Invio del Checkpoint Fotografico (Multipart Upload)
elements.uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (elements.photoFile.files.length === 0) {
        showMessage(elements.uploadMessage, 'error', 'Seleziona un file prima di inviare.');
        return;
    }

    const formData = new FormData();
    formData.append('file', elements.photoFile.files[0]);
    
    const uploadDateVal = elements.uploadDate.value;
    
    showMessage(elements.uploadMessage, 'loading', 'Allineamento e analisi in corso…');
    elements.submitUploadBtn.disabled = true;

    try {
        const response = await fetch(`/api/v1/photos/upload?entry_date=${uploadDateVal}`, {
            method: 'POST',
            body: formData
        });

        if (response.status === 201) {
            showMessage(elements.uploadMessage, 'success', 'Foto allineata e analizzata con successo!');
            elements.photoFile.value = '';
            elements.fileInfoText.textContent = "Nessun file selezionato (JPEG, PNG, HEIC)";
            
            // Ricarica tutti i dati dell'applicazione
            await loadAppData();
        } else {
            const errData = await response.json();
            const errCode = errData.detail;
            
            let userFriendlyMsg = "Errore durante l'allineamento della foto.";
            if (errCode === "NO_FACE_DETECTED") {
                userFriendlyMsg = "Qualità non conforme: Nessun volto rilevato nella foto. Assicurati che il viso sia ben visibile e illuminato.";
            } else if (errCode === "MULTIPLE_FACES_DETECTED") {
                userFriendlyMsg = "Qualità non conforme: Rilevato più di un volto. La foto deve contenere un singolo soggetto.";
            } else if (errCode === "EXCESSIVE_HEAD_TILT") {
                userFriendlyMsg = "Qualità non conforme: Inclinazione della testa eccessiva (>30°). Mantieni la testa dritta e guarda l'obiettivo.";
            } else if (errCode === "INVALID_IMAGE_PAYLOAD") {
                userFriendlyMsg = "Il file caricato è corrotto o non è un'immagine valida.";
            } else if (errCode === "DATABASE_BUSY") {
                userFriendlyMsg = "Il database è temporaneamente occupato. Riprova tra qualche istante.";
            }
            
            showMessage(elements.uploadMessage, 'error', userFriendlyMsg);
        }
    } catch (error) {
        showMessage(elements.uploadMessage, 'error', 'Errore di rete durante il caricamento della foto.');
    } finally {
        elements.submitUploadBtn.disabled = false;
    }
});

/**
 * Mostra un messaggio di feedback all'utente.
 */
function showMessage(el, type, text) {
    el.textContent = text;
    el.className = `message ${type}`;
    
    if (type !== 'loading') {
        setTimeout(() => {
            if (el.textContent === text) {
                el.style.display = 'none';
            }
        }, 6000);
    }
}

// Avvio dell'applicazione
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await loadRoutineConfig();
    } catch (err) {
        console.error('Impossibile caricare routine-config:', err);
    }
    loadAppData();
});
