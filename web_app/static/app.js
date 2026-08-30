/**
 * ECG Multimodal Representation Cockpit "” Interactive Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    let allRecords = [];
    let currentRecordId = "REC-1";
    let activeCategoryFilter = "ALL";
    let currentPage = 1;
    const PAGE_SIZE = 20;
    let radarChart = null;
    let embeddings3DData = null;

    // DOM Elements
    const plotlyContainer = document.getElementById("plotly3DCluster");
    const recordsTableBody = document.getElementById("recordsTableBody");
    const recordSearchInput = document.getElementById("recordSearchInput");
    const categoryFilterPills = document.getElementById("categoryFilterPills");
    const btnScrollToRecord = document.getElementById("btnScrollToRecord");
    const btnRegenLLM = document.getElementById("btnRegenLLM");
    const sampleSelectHeader = document.getElementById("sampleSelectHeader");

    const btnPrevPage = document.getElementById("btnPrevPage");
    const btnNextPage = document.getElementById("btnNextPage");
    const paginationInfo = document.getElementById("paginationInfo");

    const activeRecordTitle = document.getElementById("activeRecordTitle");
    const activeRecordTag = document.getElementById("activeRecordTag");
    const activeRecordMeta = document.getElementById("activeRecordMeta");
    const confidenceTableBody = document.getElementById("confidenceTableBody");
    const llmReportContent = document.getElementById("llmReportContent");
    const biomarkerTableBody = document.getElementById("biomarkerTableBody");
    const gradcamGrid = document.getElementById("gradcamGrid");

    const rawEcgCanvas = document.getElementById("rawEcgCanvas");
    const temporalAttrCanvas = document.getElementById("temporalAttrCanvas");
    const translatedWaveformCanvas = document.getElementById("translatedWaveformCanvas");

    const LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

    // Initialize Cockpit
    init();

    async function init() {
        setupEventListeners();
        if (plotlyContainer) {
            await load3DCluster();
        }
        if (recordsTableBody || sampleSelectHeader) {
            await loadRecordsCatalog();
        }
        if (allRecords.length > 0) {
            selectRecord(allRecords[0].id);
        }
    }

    function setupEventListeners() {
        if (recordSearchInput) {
            recordSearchInput.addEventListener("input", () => {
                currentPage = 1;
                filterAndRenderTable();
            });
        }

        if (categoryFilterPills) {
            categoryFilterPills.addEventListener("click", (e) => {
                if (e.target.classList.contains("pill")) {
                    document.querySelectorAll("#categoryFilterPills .pill").forEach(p => p.classList.remove("active"));
                    e.target.classList.add("active");
                    activeCategoryFilter = e.target.getAttribute("data-cat");
                    currentPage = 1;
                    filterAndRenderTable();
                }
            });
        }

        if (btnPrevPage) {
            btnPrevPage.addEventListener("click", () => {
                if (currentPage > 1) {
                    currentPage--;
                    filterAndRenderTable();
                }
            });
        }

        if (btnNextPage) {
            btnNextPage.addEventListener("click", () => {
                currentPage++;
                filterAndRenderTable();
            });
        }

        if (sampleSelectHeader) {
            sampleSelectHeader.addEventListener("change", (e) => {
                selectRecord(e.target.value);
            });
        }

        if (btnScrollToRecord) {
            btnScrollToRecord.addEventListener("click", () => {
                const section = document.getElementById("activeRecordSection") || document.getElementById("sampleDetailCard");
                if (section) section.scrollIntoView({ behavior: 'smooth' });
            });
        }

        if (btnRegenLLM) {
            btnRegenLLM.addEventListener("click", () => {
                if (currentRecordId) {
                    generateLLMInterpretation(currentRecordId);
                }
            });
        }

        const manifoldModeSelect = document.getElementById("manifoldModeSelect");
        const latentPillTag = document.getElementById("latentPillTag");
        if (manifoldModeSelect) {
            manifoldModeSelect.addEventListener("change", (e) => {
                const mode = e.target.value;
                if (latentPillTag) {
                    latentPillTag.textContent = mode === "tsne" ? "z_fused âˆˆ â„Â¹â°âµâ¶ â†’ t-SNE 3D" : "z_fused âˆˆ â„Â¹â°âµâ¶ â†’ PCA 3D";
                }
                renderPlotly3D();
            });
        }

        window.addEventListener("resize", () => {
            if (plotlyContainer && window.Plotly) Plotly.Plots.resize(plotlyContainer);
            if (currentRecordId) selectRecord(currentRecordId);
        });
    }

    async function load3DCluster() {
        try {
            const res = await fetch("/api/embeddings_3d");
            embeddings3DData = await res.json();
            renderPlotly3D();
        } catch (err) {
            console.error("Failed to load 3D embeddings:", err);
        }
    }

    function renderPlotly3D() {
        if (!plotlyContainer || !window.Plotly || !embeddings3DData) return;

        const manifoldModeSelect = document.getElementById("manifoldModeSelect");
        const mode = manifoldModeSelect ? manifoldModeSelect.value : "tsne";
        const points = (mode === "pca" && embeddings3DData.population_points_pca) ? 
            embeddings3DData.population_points_pca : 
            (embeddings3DData.population_points_tsne || embeddings3DData.population_points || []);

        const categories = ["NORM", "MI", "STTC", "CD", "HYP"];
        const colorMap = {
            NORM: "#10b981",
            MI: "#f43f5e",
            STTC: "#f59e0b",
            CD: "#a855f7",
            HYP: "#3b82f6"
        };

        const traces = [];

        categories.forEach(cat => {
            const catPts = points.filter(p => p.category === cat);
            traces.push({
                x: catPts.map(p => p.x),
                y: catPts.map(p => p.y),
                z: catPts.map(p => p.z),
                text: catPts.map(p => `Record #${p.ecg_id || p.sample_code} (${p.category})`),
                customdata: catPts.map(p => `REC-${p.ecg_id}`),
                mode: 'markers',
                type: 'scatter3d',
                name: cat,
                marker: {
                    size: 4.8,
                    color: colorMap[cat],
                    opacity: 0.85
                }
            });
        });

        // Active Record Highlight Marker
        const activeRecord = allRecords.find(r => r.id === currentRecordId);
        const activeCoords = activeRecord ? (activeRecord[`coords_3d_${mode}`] || activeRecord.coords_3d) : null;
        if (activeCoords) {
            traces.push({
                x: [activeCoords.x],
                y: [activeCoords.y],
                z: [activeCoords.z],
                text: [`â˜… ACTIVE: ${activeRecord.name}`],
                mode: 'markers',
                type: 'scatter3d',
                name: 'Active Patient',
                marker: {
                    size: 11,
                    color: '#ffffff',
                    symbol: 'diamond',
                    line: { color: '#06b6d4', width: 3 }
                }
            });
        }

        const axisLabelPrefix = mode === "tsne" ? "t-SNE" : "PCA";
        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            scene: {
                xaxis: { title: `${axisLabelPrefix} 1`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                yaxis: { title: `${axisLabelPrefix} 2`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                zaxis: { title: `${axisLabelPrefix} 3`, gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                bgcolor: 'transparent',
                camera: { eye: { x: 1.4, y: 1.4, z: 1.1 } }
            },
            legend: {
                font: { color: '#94a3b8', family: 'Inter', size: 10 },
                bgcolor: 'rgba(15,23,42,0.8)'
            }
        };

        const config = { responsive: true, displayModeBar: false };
        Plotly.newPlot(plotlyContainer, traces, layout, config);

        plotlyContainer.on('plotly_click', (data) => {
            if (data.points && data.points.length > 0) {
                const pt = data.points[0];
                if (pt.customdata) {
                    selectRecord(pt.customdata, true);
                }
            }
        });
    }

    function getDisplayHR(r) {
        if (!r) return 75;
        // Priority 1: Check biomarkers.RR_Mean which contains the measured cardiac cycle
        if (r.biomarkers && r.biomarkers.RR_Mean) {
            let rr = Number(r.biomarkers.RR_Mean);
            if (rr > 0.25 && rr < 3.0) return Math.round(60.0 / rr);
            if (rr >= 250 && rr <= 3000) return Math.round(60000.0 / rr);
        }
        let hr = Number(r.heart_rate);
        if (!isNaN(hr) && hr > 0) {
            if (hr >= 30 && hr <= 220) return Math.round(hr);
            // If hr is scaled by 1000 (e.g., 76775 -> 76.775 -> 77)
            if (hr >= 30000 && hr <= 220000) return Math.round(hr / 1000.0);
            if (hr > 220 && hr < 3000) return Math.round(60000.0 / hr);
        }
        return 75;
    }

    let isFirstLoad = true;

    async function loadRecordsCatalog() {
        if (!recordsTableBody) return;
        
        const query = recordSearchInput ? recordSearchInput.value.trim() : "";
        const cat = activeCategoryFilter;
        
        try {
            const res = await fetch(`/api/records?page=${currentPage}&limit=${PAGE_SIZE}&category=${cat}&search=${encodeURIComponent(query)}`);
            const data = await res.json();
            
            allRecords = data.records || [];
            const totalCount = data.total_count || 0;
            const totalPages = data.total_pages || 1;
            
            // Render table rows for the 20 records
            recordsTableBody.innerHTML = "";
            allRecords.forEach(r => {
                const tr = document.createElement("tr");
                if (r.id === currentRecordId) tr.classList.add("active-row");

                const hrVal = getDisplayHR(r);
                tr.innerHTML = `
                    <td><strong>${r.sample_code || r.id}</strong></td>
                    <td>${r.age}yo ${r.sex}</td>
                    <td><span class="patient-tag tag-${r.category.toLowerCase()}">${r.category}</span></td>
                    <td>${r.clinical_history || 'Routine 12-lead study'}</td>
                    <td>${hrVal} bpm</td>
                    <td><button class="btn-run-pipeline" data-id="${r.id}">▶ Run Pipeline</button></td>
                `;

                tr.addEventListener("click", () => selectRecord(r.id, true));
                recordsTableBody.appendChild(tr);
            });

                        // Update Pagination Controls
            if (paginationInfo) {
                if (totalCount === 0) {
                    paginationInfo.textContent = "0 records found";
                } else {
                    const startIndex = (currentPage - 1) * PAGE_SIZE;
                    const endIndex = Math.min(startIndex + PAGE_SIZE, totalCount);
                    paginationInfo.textContent = `Page ${currentPage} of ${totalPages} (${startIndex + 1}–${endIndex} of ${totalCount})`;
                }
            }
            if (btnPrevPage) {
                btnPrevPage.disabled = (currentPage <= 1);
            }
            if (btnNextPage) {
                btnNextPage.disabled = (currentPage >= totalPages);
            }

            if (isFirstLoad && allRecords.length > 0) {
                isFirstLoad = false;
                selectRecord(allRecords[0].id, false);
            }
            
            // Update sample select in header
            if (sampleSelectHeader) {
                sampleSelectHeader.innerHTML = "";
                allRecords.forEach(r => {
                    const opt = document.createElement("option");
                    opt.value = r.id;
                    opt.textContent = `${r.sample_code || r.id}: ${r.category} "” ${r.name}`;
                    if (r.id === currentRecordId) opt.selected = true;
                    sampleSelectHeader.appendChild(opt);
                });
            }
        } catch (err) {
            console.error("Failed to load records catalog:", err);
        }
    }

    function filterAndRenderTable() {
        loadRecordsCatalog();
    }

    async function selectRecord(recordId, shouldScroll = false) {
        currentRecordId = recordId;

        if (recordsTableBody) {
            document.querySelectorAll("#recordsTableBody tr").forEach(row => row.classList.remove("active-row"));
            const activeTr = Array.from(recordsTableBody.querySelectorAll("tr")).find(tr => tr.innerHTML.includes(recordId));
            if (activeTr) activeTr.classList.add("active-row");
        }

        if (sampleSelectHeader) {
            sampleSelectHeader.value = recordId;
        }

        if (plotlyContainer) {
            renderPlotly3D();
        }

        try {
            const res = await fetch(`/api/sample/${recordId}`);
            const data = await res.json();
            if (data.status === "success" && data.payload) {
                renderRecordDetail(data.payload);
                if (shouldScroll) {
                    const section = document.getElementById("activeRecordSection") || document.getElementById("sampleDetailCard");
                    if (section) section.scrollIntoView({ behavior: 'smooth' });
                }
            }
        } catch (err) {
            console.error("Failed to load record details:", err);
        }
    }

    function renderRecordDetail(payload) {
        const record = payload.record;
        
        // Header
        if (activeRecordTitle) activeRecordTitle.textContent = `Patient Case — ${record.name || record.id}`;
        if (activeRecordTag) {
            activeRecordTag.textContent = `DIAGNOSTIC CLASS: ${record.category}`;
            activeRecordTag.className = `patient-tag tag-${record.category.toLowerCase()}`;
        }
        if (activeRecordMeta) {
            const displayHR = getDisplayHR(record);
            activeRecordMeta.textContent = `${record.age}yo ${record.sex}  ·  Heart Rate: ${displayHR} bpm  ·  Ground Truth: ${(record.ground_truth || []).join(', ')}`;
        }

        // Confidence Table
        renderConfidenceTable(payload.model_confidences);

        // 1. Raw Waveforms
        if (rawEcgCanvas) draw12LeadWaveforms(rawEcgCanvas, payload.signal, false, [], `Actual ECG - ${record.category}`);

        // 2. Temporal Integrated Gradients Attribution
        if (temporalAttrCanvas) draw12LeadWaveforms(temporalAttrCanvas, payload.signal, true, payload.temporal_attributions, `12-Lead Temporal Integrated Gradients Attribution (${record.category})`);

        // 3. Morphology 2D Grad-CAM Grid
        if (gradcamGrid) renderGradCAMGrid(payload.morphology_gradcams);

        // 4. Translated Waveform Attribution
        if (translatedWaveformCanvas) drawTranslatedWaveform(translatedWaveformCanvas, payload.signal, payload.translated_boxes);

        // 5. Biomarkers
        if (biomarkerTableBody) renderBiomarkers(payload.biomarkers || {});

        // 6. Gemini LLM Synthesis
        generateLLMInterpretation(record.id, payload);
    }

    function renderConfidenceTable(conf) {
        if (!confidenceTableBody) return;
        confidenceTableBody.innerHTML = "";
        for (const [modelName, probs] of Object.entries(conf || {})) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${modelName}</strong></td>
                <td>${(probs.NORM || 0).toFixed(2)}%</td>
                <td>${(probs.MI || 0).toFixed(2)}%</td>
                <td>${(probs.STTC || 0).toFixed(2)}%</td>
                <td>${(probs.CD || 0).toFixed(2)}%</td>
                <td>${(probs.HYP || 0).toFixed(2)}%</td>
            `;
            confidenceTableBody.appendChild(tr);
        }
    }

    function draw12LeadWaveforms(canvas, signals, overlayAttribution = false, attributions = [], title = "") {
        if (!canvas || !signals || signals.length < 12) return;
        const ctx = canvas.getContext("2d");
        const rect = canvas.getBoundingClientRect();
        const displayHeight = 980;
        
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = displayHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        const width = rect.width;
        const height = displayHeight;

        // Clean White Medical Publication Canvas
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, width, height);

        const marginTop = 35;
        const marginBottom = 45;
        const marginLeft = 65;
        const marginRight = 30;
        const plotWidth = width - marginLeft - marginRight;
        const totalPlotHeight = height - marginTop - marginBottom;
        const leadHeight = totalPlotHeight / 12;
        const numSamples = signals[0].length;

        // Top Title if provided
        if (title) {
            ctx.fillStyle = "#1e293b";
            ctx.font = "bold 13px Inter";
            ctx.textAlign = "center";
            ctx.fillText(title, width / 2, 22);
            ctx.textAlign = "left";
        }

        for (let l = 0; l < 12; l++) {
            const leadSignal = signals[l];
            const leadAttr = (overlayAttribution && attributions[l]) ? attributions[l] : null;
            const topY = marginTop + l * leadHeight;
            const bottomY = topY + leadHeight;
            const midY = topY + leadHeight / 2;

            // Subplot horizontal divider / bottom spine
            ctx.strokeStyle = "#e2e8f0";
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            ctx.moveTo(marginLeft, bottomY);
            ctx.lineTo(marginLeft + plotWidth, bottomY);
            ctx.stroke();

            // Find lead peak voltage for proper dynamic scaling
            let maxAbs = 0.25;
            for (let s = 0; s < numSamples; s++) {
                if (Math.abs(leadSignal[s]) > maxAbs) maxAbs = Math.abs(leadSignal[s]);
            }
            maxAbs = Math.max(0.3, maxAbs * 1.15);

            // Isoelectric 0 mV Baseline (dotted)
            const zeroY = midY;
            ctx.strokeStyle = "#cbd5e1";
            ctx.lineWidth = 0.8;
            ctx.setLineDash([3, 4]);
            ctx.beginPath();
            ctx.moveTo(marginLeft, zeroY);
            ctx.lineTo(marginLeft + plotWidth, zeroY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Lead Name (Bold on left)
            ctx.fillStyle = "#0f172a";
            ctx.font = "bold 11px Inter";
            ctx.textAlign = "right";
            ctx.fillText(LEAD_NAMES[l], marginLeft - 26, midY + 4);

            // Voltage Ticks (min, zero, max)
            ctx.fillStyle = "#64748b";
            ctx.font = "9px JetBrains Mono";
            ctx.fillText(`+${maxAbs.toFixed(1)}`, marginLeft - 4, topY + 10);
            ctx.fillText(" 0.0", marginLeft - 4, zeroY + 3);
            ctx.fillText(`-${maxAbs.toFixed(1)}`, marginLeft - 4, bottomY - 3);
            ctx.textAlign = "left";

            // Map sample coordinates
            const scaleY = (leadHeight * 0.42) / maxAbs;
            const dx = plotWidth / (numSamples - 1);

            // Draw ECG Trace with authentic noise and resolution
            for (let i = 0; i < numSamples - 1; i++) {
                const x1 = marginLeft + i * dx;
                const y1 = zeroY - (leadSignal[i] * scaleY);
                const x2 = marginLeft + (i + 1) * dx;
                const y2 = zeroY - (leadSignal[i + 1] * scaleY);

                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);

                if (overlayAttribution && leadAttr) {
                    const attrVal = (leadAttr[i] + leadAttr[i + 1]) / 2;
                    if (attrVal > 0.45) {
                        ctx.strokeStyle = `rgba(239, 35, 60, ${Math.min(1.0, 0.7 + attrVal * 0.3)})`;
                        ctx.lineWidth = 2.4;
                    } else if (attrVal > 0.22) {
                        ctx.strokeStyle = `rgba(245, 158, 11, ${0.6 + attrVal * 0.4})`;
                        ctx.lineWidth = 1.8;
                    } else {
                        ctx.strokeStyle = "rgba(43, 45, 66, 0.45)";
                        ctx.lineWidth = 1.1;
                    }
                } else {
                    ctx.strokeStyle = "#2B2D42";
                    ctx.lineWidth = 1.35;
                }
                ctx.stroke();
            }
        }

        // Bottom X-Axis Ticks & Label
        const bottomAxisY = marginTop + totalPlotHeight;
        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(marginLeft, bottomAxisY);
        ctx.lineTo(marginLeft + plotWidth, bottomAxisY);
        ctx.stroke();

        const xTicks = [0, 200, 400, 600, 800, 1000];
        ctx.fillStyle = "#334155";
        ctx.font = "bold 9px JetBrains Mono";
        xTicks.forEach(tick => {
            const x = marginLeft + (tick / 1000) * plotWidth;
            ctx.beginPath();
            ctx.moveTo(x, bottomAxisY);
            ctx.lineTo(x, bottomAxisY + 5);
            ctx.stroke();
            ctx.textAlign = "center";
            ctx.fillText(tick.toString(), x, bottomAxisY + 16);
        });

        // X-Axis Title
        ctx.font = "bold 10px Inter";
        ctx.fillText("Time steps (Samples)", marginLeft + plotWidth / 2, bottomAxisY + 32);
        ctx.textAlign = "left";
    }

    function renderGradCAMGrid(gradcams) {
        if (!gradcamGrid) return;
        gradcamGrid.innerHTML = "";
        if (!gradcams || gradcams.length < 12) return;

        for (let l = 0; l < 12; l++) {
            const card = document.createElement("div");
            card.className = "gradcam-card";

            const title = document.createElement("div");
            title.className = "gradcam-lead-name";
            title.textContent = `Lead ${LEAD_NAMES[l]} Grad-CAM`;

            const canvas = document.createElement("canvas");
            canvas.className = "gradcam-canvas";
            card.appendChild(title);
            card.appendChild(canvas);
            gradcamGrid.appendChild(card);

            drawGradCAMHeatmap(canvas, gradcams[l]);
        }
    }

    function drawGradCAMHeatmap(canvas, matrix) {
        if (!canvas || !matrix) return;
        const ctx = canvas.getContext("2d");
        const rows = matrix.length;
        const cols = matrix[0].length;

        canvas.width = 120;
        canvas.height = 70;

        const cellW = canvas.width / cols;
        const cellH = canvas.height / rows;

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const v = matrix[r][c];
                const red = Math.min(255, Math.floor(v * 320));
                const green = Math.min(255, Math.floor((1 - Math.abs(v - 0.5) * 2) * 255));
                const blue = Math.min(255, Math.floor((1 - v) * 255));

                ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
                ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
            }
        }
    }

    function drawTranslatedWaveform(canvas, signals, boxes) {
        if (!canvas || !signals) return;
        draw12LeadWaveforms(canvas, signals, false, [], "Lead-Specific ECG Grad-CAM Attribution & Landmark Mapping");

        const ctx = canvas.getContext("2d");
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const displayHeight = 980;
        const marginTop = 35;
        const marginBottom = 45;
        const marginLeft = 65;
        const marginRight = 30;
        const plotWidth = width - marginLeft - marginRight;
        const totalPlotHeight = displayHeight - marginTop - marginBottom;
        const leadHeight = totalPlotHeight / 12;
        const numSamples = signals[0].length;
        const dx = plotWidth / (numSamples - 1);

        (boxes || []).forEach(box => {
            const l = box.lead || 1;
            const topY = marginTop + l * leadHeight;
            const bottomY = topY + leadHeight;
            const startX = marginLeft + box.start * dx;
            const endX = marginLeft + box.end * dx;
            const boxWidth = Math.max(16, endX - startX);

            // Coral Highlight Span
            ctx.fillStyle = "rgba(239, 35, 60, 0.16)";
            ctx.fillRect(startX, topY + 4, boxWidth, leadHeight - 8);

            // Dashed boundary lines
            ctx.strokeStyle = "#EF233C";
            ctx.lineWidth = 1.0;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(startX, topY + 4);
            ctx.lineTo(startX, bottomY - 4);
            ctx.moveTo(endX, topY + 4);
            ctx.lineTo(endX, bottomY - 4);
            ctx.stroke();
            ctx.setLineDash([]);

            // Label pill badge
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(startX + 4, topY + 8, Math.min(130, boxWidth + 40), 16);
            ctx.strokeStyle = "#EF233C";
            ctx.strokeRect(startX + 4, topY + 8, Math.min(130, boxWidth + 40), 16);

            ctx.fillStyle = "#D90429";
            ctx.font = "bold 8.5px Inter";
            ctx.fillText(`${box.label} (Attr: ${box.attr_score})`, startX + 7, topY + 20);
        });
    }

    function renderBiomarkers(bio) {
        if (!biomarkerTableBody) return;
        const radarCanvas = document.getElementById("biomarkerRadarChart");

        // Helper: auto-convert seconds -> ms if value < 2.0
        function toMs(v) { return (v != null && v < 2.0) ? v * 1000 : (v ?? 0); }
        function fmt(v, dec=1) { return (v != null) ? parseFloat(v).toFixed(dec) : "—"; }

        // Extract all 24 CWT features
        const rrMean  = toMs(bio.RR_Mean);
        const qrs     = toMs(bio.QRS_Duration);
        const pr      = toMs(bio.PR_Interval);
        const qt      = toMs(bio.QT_Interval);
        const qtc     = toMs(bio.QTc_Bazett);
        const stDur   = toMs(bio.ST_Duration);
        const pDur    = toMs(bio.P_wave_Duration);
        const rAmp    = bio.R_Amplitude ?? null;
        const pAmp    = bio.P_Amplitude ?? null;
        const tAmp    = bio.T_Amplitude ?? null;
        const stDev   = bio.ST_Deviation ?? null;
        const qAmp    = bio.Q_Amplitude ?? null;
        const rsRatio = bio.R_S_Ratio ?? null;
        const qrsEng  = bio.QRS_Energy ?? null;
        const sdnn    = toMs(bio.SDNN);
        const rmssd   = toMs(bio.RMSSD);
        const pnn50   = bio.pNN50 ?? null;
        const pnn20   = bio.pNN20 ?? null;
        const sdrr    = bio.SDRR_RMSSD_Ratio ?? null;
        const hrvTri  = bio.HRV_Triangular_Index ?? null;
        const lfPow   = bio.LF_Power ?? null;
        const hfPow   = bio.HF_Power ?? null;
        const lfhf    = bio.LF_HF_Ratio ?? null;
        const totPow  = bio.Total_Power ?? null;

        // 8-axis Radar (representative clinical axes from the 24 CWT features)
        const radarLabels = ["RR Mean", "QRS Dur", "PR Interval", "QTc", "ST Dev", "R Amp", "SDNN", "LF/HF"];
        const normalize = (v, lo, hi) => Math.max(5, Math.min(100, ((v - lo) / (hi - lo)) * 100));
        const patientVals = [
            normalize(rrMean, 600, 1100),
            normalize(qrs, 60, 130),
            normalize(pr, 100, 220),
            normalize(qtc, 340, 480),
            normalize((stDev ?? 0) + 0.15, 0, 0.30),
            normalize(rAmp ?? 1.0, 0, 2.5),
            normalize(sdnn, 10, 90),
            normalize(lfhf ?? 1.0, 0, 4.0),
        ];

        if (radarCanvas && window.Chart) {
            const ctx = radarCanvas.getContext("2d");
            if (radarChart) radarChart.destroy();
            radarChart = new Chart(ctx, {
                type: "radar",
                data: {
                    labels: radarLabels,
                    datasets: [
                        {
                            label: "Patient ECG Features",
                            data: patientVals,
                            backgroundColor: "rgba(6, 182, 212, 0.25)",
                            borderColor: "#06b6d4",
                            pointBackgroundColor: "#06b6d4",
                            borderWidth: 2
                        },
                        {
                            label: "Reference Baseline Envelope",
                            data: [50, 50, 50, 50, 50, 50, 50, 50],
                            backgroundColor: "rgba(255, 255, 255, 0.05)",
                            borderColor: "rgba(255, 255, 255, 0.3)",
                            borderWidth: 1,
                            borderDash: [4, 4]
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        r: {
                            angleLines: { color: "rgba(255, 255, 255, 0.1)" },
                            grid: { color: "rgba(255, 255, 255, 0.08)" },
                            ticks: { display: false },
                            pointLabels: { color: "#94a3b8", font: { family: "Inter", size: 11 } },
                            suggestedMin: 0,
                            suggestedMax: 100
                        }
                    },
                    plugins: {
                        legend: { labels: { color: "#94a3b8", font: { family: "Inter", size: 11 } } }
                    }
                }
            });
        }

        // Clinical evaluation helpers
        const ok   = t => `<span class="text-success">${t}</span>`;
        const warn = t => `<span class="text-warning">${t}</span>`;
        const bad  = t => `<span class="text-danger">${t}</span>`;

        // Interval evaluations
        const qrsEval  = qrs > 120 ? bad("Prolonged — Conduction Delay") : qrs < 70 ? warn("Narrow QRS") : ok("Normal (70–110 ms)");
        const prEval   = pr > 200 ? bad("1st Degree AV Block") : pr < 110 ? warn("Short PR — Pre-excitation") : ok("Normal (120–200 ms)");
        const qtEval   = qt > 450 ? bad("Prolonged QT") : qt < 330 ? warn("Short QT") : ok("Normal (350–450 ms)");
        const qtcEval  = qtc > 460 ? bad("Prolonged QTc ≥ 460 ms") : qtc < 350 ? warn("Short QTc") : ok("Normal (360–440 ms)");
        const rrEval   = rrMean < 600 ? warn("Tachycardia") : rrMean > 1200 ? warn("Bradycardia") : ok("Normal Sinus Rhythm");
        const stDurEval = stDur > 120 ? warn("Prolonged ST") : ok("Normal");
        const pDurEval = pDur > 120 ? bad("Atrial Overload") : pDur < 60 ? warn("Short P-wave") : ok("Normal (60–120 ms)");
        // Amplitude evaluations
        const rAmpEval = rAmp == null ? "—" : rAmp > 2.5 ? warn("High R — LVH?") : rAmp < 0.3 ? warn("Low Voltage") : ok("Normal");
        const pAmpEval = pAmp == null ? "—" : pAmp > 0.25 ? warn("Tall P — P Pulmonale?") : ok("Normal");
        const tAmpEval = tAmp == null ? "—" : tAmp < 0 ? bad("T-wave Inversion") : tAmp > 1.0 ? warn("Hyperacute T") : ok("Normal");
        const stDevEval = stDev == null ? "—" : stDev > 0.10 ? bad(`ST-Elevation +${fmt(stDev,2)} mV`) : stDev < -0.08 ? bad(`ST-Depression ${fmt(stDev,2)} mV`) : ok("Isoelectric");
        const qAmpEval = qAmp == null ? "—" : Math.abs(qAmp) > 0.3 ? bad("Pathological Q-wave") : ok("Normal");
        const rsEval   = rsRatio == null ? "—" : rsRatio > 1 ? ok("Normal (R > S)") : warn("R/S < 1 — Anterior Dominance?");
        // HRV evaluations
        const sdnnEval  = sdnn < 25 ? bad("Severely Depressed HRV") : sdnn < 40 ? warn("Reduced HRV") : ok("Intact Autonomic Tone");
        const rmssdEval = rmssd == null ? "—" : rmssd < 15 ? bad("Parasympathetic Withdrawal") : rmssd < 30 ? warn("Reduced") : ok("Normal (≥ 30 ms)");
        const pnn50Eval = pnn50 == null ? "—" : pnn50 < 3 ? warn("Low pNN50") : ok("Normal");
        const pnn20Eval = pnn20 == null ? "—" : pnn20 < 10 ? warn("Low pNN20") : ok("Normal");
        const sdrEval   = sdrr == null ? "—" : ok("Computed");
        const hrvTriEval = hrvTri == null ? "—" : hrvTri < 10 ? bad("Reduced") : ok("Normal (≥ 10)");
        const lfhfEval  = lfhf == null ? "—" : lfhf > 3.0 ? warn("Sympathetic Dominance") : lfhf < 0.5 ? warn("Parasympathetic Dominance") : ok("Balanced ANS");

        biomarkerTableBody.innerHTML = `
            <tr class="table-section-header"><td colspan="4" style="color:#06b6d4;font-size:11px;padding:6px 8px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8">⏱ Interval Features</td></tr>
            <tr><td><strong>RR Mean</strong></td><td>${fmt(rrMean)} ms</td><td>600 – 1000 ms</td><td>${rrEval}</td></tr>
            <tr><td><strong>QRS Duration</strong></td><td>${fmt(qrs)} ms</td><td>70 – 110 ms</td><td>${qrsEval}</td></tr>
            <tr><td><strong>PR Interval</strong></td><td>${fmt(pr)} ms</td><td>120 – 200 ms</td><td>${prEval}</td></tr>
            <tr><td><strong>QT Interval</strong></td><td>${fmt(qt)} ms</td><td>350 – 450 ms</td><td>${qtEval}</td></tr>
            <tr><td><strong>QTc (Bazett)</strong></td><td>${fmt(qtc)} ms</td><td>360 – 440 ms</td><td>${qtcEval}</td></tr>
            <tr><td><strong>ST Duration</strong></td><td>${fmt(stDur)} ms</td><td>80 – 120 ms</td><td>${stDurEval}</td></tr>
            <tr><td><strong>P-wave Duration</strong></td><td>${fmt(pDur)} ms</td><td>60 – 120 ms</td><td>${pDurEval}</td></tr>
            <tr class="table-section-header"><td colspan="4" style="color:#a78bfa;font-size:11px;padding:6px 8px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8">📐 Amplitude Features</td></tr>
            <tr><td><strong>R Amplitude</strong></td><td>${rAmp != null ? fmt(rAmp,3)+" mV" : "—"}</td><td>0.5 – 2.0 mV</td><td>${rAmpEval}</td></tr>
            <tr><td><strong>P Amplitude</strong></td><td>${pAmp != null ? fmt(pAmp,3)+" mV" : "—"}</td><td>0.05 – 0.25 mV</td><td>${pAmpEval}</td></tr>
            <tr><td><strong>T Amplitude</strong></td><td>${tAmp != null ? fmt(tAmp,3)+" mV" : "—"}</td><td>0.1 – 0.8 mV</td><td>${tAmpEval}</td></tr>
            <tr><td><strong>ST-Segment Deviation</strong></td><td>${stDev != null ? (stDev>=0?"+":"")+fmt(stDev,3)+" mV" : "—"}</td><td>-0.05 – +0.10 mV</td><td>${stDevEval}</td></tr>
            <tr><td><strong>Q Amplitude</strong></td><td>${qAmp != null ? fmt(qAmp,3)+" mV" : "—"}</td><td>&lt; 0.3 mV</td><td>${qAmpEval}</td></tr>
            <tr><td><strong>R/S Ratio</strong></td><td>${rsRatio != null ? fmt(rsRatio,2) : "—"}</td><td>&gt; 1.0 (lateral)</td><td>${rsEval}</td></tr>
            <tr><td><strong>QRS Energy</strong></td><td>${qrsEng != null ? fmt(qrsEng,4) : "—"}</td><td>Signal power</td><td>${qrsEng != null ? ok("Computed") : "—"}</td></tr>
            <tr class="table-section-header"><td colspan="4" style="color:#34d399;font-size:11px;padding:6px 8px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8">🫀 HRV Time-Domain</td></tr>
            <tr><td><strong>SDNN</strong></td><td>${fmt(sdnn)} ms</td><td>&gt; 40 ms</td><td>${sdnnEval}</td></tr>
            <tr><td><strong>RMSSD</strong></td><td>${fmt(rmssd)} ms</td><td>&gt; 30 ms</td><td>${rmssdEval}</td></tr>
            <tr><td><strong>pNN50</strong></td><td>${pnn50 != null ? fmt(pnn50,1)+" %" : "—"}</td><td>&gt; 3 %</td><td>${pnn50Eval}</td></tr>
            <tr><td><strong>pNN20</strong></td><td>${pnn20 != null ? fmt(pnn20,1)+" %" : "—"}</td><td>&gt; 10 %</td><td>${pnn20Eval}</td></tr>
            <tr><td><strong>SDRR/RMSSD Ratio</strong></td><td>${sdrr != null ? fmt(sdrr,2) : "—"}</td><td>Balance index</td><td>${sdrEval}</td></tr>
            <tr><td><strong>HRV Triangular Index</strong></td><td>${hrvTri != null ? fmt(hrvTri,2) : "—"}</td><td>&gt; 10</td><td>${hrvTriEval}</td></tr>
            <tr class="table-section-header"><td colspan="4" style="color:#fb923c;font-size:11px;padding:6px 8px;letter-spacing:0.08em;text-transform:uppercase;opacity:0.8">📡 HRV Frequency-Domain</td></tr>
            <tr><td><strong>LF Power</strong></td><td>${lfPow != null ? fmt(lfPow,2)+" ms²" : "—"}</td><td>0.04 – 0.15 Hz band</td><td>${lfPow != null ? ok("Recorded") : "—"}</td></tr>
            <tr><td><strong>HF Power</strong></td><td>${hfPow != null ? fmt(hfPow,2)+" ms²" : "—"}</td><td>0.15 – 0.40 Hz band</td><td>${hfPow != null ? ok("Recorded") : "—"}</td></tr>
            <tr><td><strong>LF/HF Ratio</strong></td><td>${lfhf != null ? fmt(lfhf,2) : "—"}</td><td>0.5 – 2.0</td><td>${lfhfEval}</td></tr>
            <tr><td><strong>Total HRV Power</strong></td><td>${totPow != null ? fmt(totPow,2)+" ms²" : "—"}</td><td>Spectral sum</td><td>${totPow != null ? ok("Recorded") : "—"}</td></tr>
        `;
    }
    async function generateLLMInterpretation(recordId, payload) {
        if (!llmReportContent) return;
        llmReportContent.innerHTML = `
            <div class="llm-loading-state">
                <span class="spinner"></span>
                <p>Synthesizing deep cardiological report from multimodal representations via Gemini API...</p>
            </div>
        `;

        try {
            const record = (payload && payload.record) ? payload.record : allRecords.find(r => r.id === recordId);
            const probs = (payload && payload.model_confidences && payload.model_confidences["Fusion (Joint)"]) ? payload.model_confidences["Fusion (Joint)"] : { NORM: 20, MI: 20, STTC: 20, CD: 20, HYP: 20 };

            const rawBio = (payload && payload.biomarkers) ? payload.biomarkers : {};
            const cleanBio = {
                QRS_Duration_ms: ((rawBio.QRS_Duration ?? 0.09) < 2.0 ? (rawBio.QRS_Duration ?? 0.09) * 1000 : (rawBio.QRS_Duration ?? 90)).toFixed(1) + " ms",
                QTc_Bazett_ms: ((rawBio.QTc_Bazett ?? 0.41) < 2.0 ? (rawBio.QTc_Bazett ?? 0.41) * 1000 : (rawBio.QTc_Bazett ?? 410)).toFixed(1) + " ms",
                PR_Interval_ms: ((rawBio.PR_Interval ?? 0.16) < 2.0 ? (rawBio.PR_Interval ?? 0.16) * 1000 : (rawBio.PR_Interval ?? 160)).toFixed(1) + " ms",
                ST_Deviation_mV: (rawBio.ST_Deviation ?? 0.0).toFixed(2) + " mV",
                SDNN_ms: ((rawBio.SDNN ?? 0.045) < 2.0 ? (rawBio.SDNN ?? 0.045) * 1000 : (rawBio.SDNN ?? 45)).toFixed(1) + " ms",
                R_Amplitude_mV: (rawBio.R_Amplitude ?? 1.1).toFixed(2) + " mV"
            };

            const res = await fetch("/api/interpret", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    record_id: record ? record.name : recordId,
                    detected_conditions: record ? record.ground_truth : ["ECG Abnormality"],
                    probabilities: {
                        NORM: (probs.NORM || 0) / 100,
                        MI: (probs.MI || 0) / 100,
                        STTC: (probs.STTC || 0) / 100,
                        CD: (probs.CD || 0) / 100,
                        HYP: (probs.HYP || 0) / 100,
                    },
                    thresholds: { NORM: 0.53, MI: 0.26, STTC: 0.25, CD: 0.30, HYP: 0.33 },
                    biomarkers: cleanBio,
                    patient_metadata: {
                        Age: record ? record.age : 60,
                        Sex: record ? record.sex : "Unknown",
                        History: record ? record.clinical_history : "ECG Study"
                    }
                })
            });

            const data = await res.json();
            if (data.report_markdown && window.marked) {
                llmReportContent.innerHTML = marked.parse(data.report_markdown);
            } else {
                llmReportContent.innerHTML = `<p class="text-danger">Failed to generate LLM consultation: ${data.message || 'Unknown error'}</p>`;
            }
        } catch (err) {
            console.error("LLM interpretation failed:", err);
            llmReportContent.innerHTML = `<p class="text-danger">Error connecting to Gemini clinical interpreter service.</p>`;
        }
    }
});
