/**
 * ECG Multimodal Representation Cockpit — Interactive Frontend Controller
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
        if (!plotlyContainer || !window.Plotly || !embeddings3DData || !embeddings3DData.population_points) return;

        const points = embeddings3DData.population_points;
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
                    size: 4.5,
                    color: colorMap[cat],
                    opacity: 0.82
                }
            });
        });

        // Active Record Highlight Marker
        const activeRecord = allRecords.find(r => r.id === currentRecordId);
        if (activeRecord && activeRecord.coords_3d) {
            traces.push({
                x: [activeRecord.coords_3d.x],
                y: [activeRecord.coords_3d.y],
                z: [activeRecord.coords_3d.z],
                text: [`★ ACTIVE: ${activeRecord.name}`],
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

        const layout = {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            scene: {
                xaxis: { title: 'PCA 1', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                yaxis: { title: 'PCA 2', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.2)' },
                zaxis: { title: 'PCA 3', gridcolor: 'rgba(255,255,255,0.08)', zerolinecolor: 'rgba(255,255,255,0.2)' },
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
                    selectRecord(pt.customdata);
                }
            }
        });
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

                tr.innerHTML = `
                    <td><strong>${r.sample_code || r.id}</strong></td>
                    <td>${r.age}yo ${r.sex}</td>
                    <td><span class="patient-tag tag-${r.category.toLowerCase()}">${r.category}</span></td>
                    <td>${r.clinical_history || 'Routine 12-lead study'}</td>
                    <td>${r.heart_rate || 75} bpm</td>
                    <td><button class="btn-select-record" data-id="${r.id}">Inspect</button></td>
                `;

                tr.addEventListener("click", () => selectRecord(r.id));
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
                selectRecord(allRecords[0].id);
            }
            
            // Update sample select in header
            if (sampleSelectHeader) {
                sampleSelectHeader.innerHTML = "";
                allRecords.forEach(r => {
                    const opt = document.createElement("option");
                    opt.value = r.id;
                    opt.textContent = `${r.sample_code || r.id}: ${r.category} — ${r.name}`;
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

    async function selectRecord(recordId) {
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
            activeRecordMeta.textContent = `${record.age}yo ${record.sex} • Heart Rate: ${record.heart_rate} bpm • Ground Truth: ${(record.ground_truth || []).join(', ')}`;
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
        const radarLabels = ["HRV SDNN", "QRS Duration", "QTc Bazett", "ST Deviation", "R Amplitude", "PR Interval"];
        const sdnn = bio.SDNN || 35;
        const qrs = bio.QRS_Duration || 90;
        const qtc = bio.QTc_Bazett || 420;
        const st = bio.ST_Deviation || 0;
        const rAmp = bio.R_Amplitude || 1.2;
        const pr = bio.PR_Interval || 160;

        const patientVals = [
            Math.min(100, (sdnn / 50) * 50),
            Math.min(100, (qrs / 120) * 50),
            Math.min(100, (qtc / 450) * 50),
            Math.min(100, 50 + st * 40),
            Math.min(100, (rAmp / 1.5) * 50),
            Math.min(100, (pr / 200) * 50),
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
                            data: [50, 50, 50, 50, 50, 50],
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

        biomarkerTableBody.innerHTML = `
            <tr>
                <td><strong>QRS Duration</strong></td>
                <td>${qrs.toFixed(1)} ms</td>
                <td>70 – 110 ms</td>
                <td><span class="${qrs > 120 ? 'text-danger' : 'text-success'}">${qrs > 120 ? 'Prolonged (Conduction Delay)' : 'Normal'}</span></td>
            </tr>
            <tr>
                <td><strong>QTc (Bazett)</strong></td>
                <td>${qtc.toFixed(1)} ms</td>
                <td>360 – 440 ms</td>
                <td><span class="${qtc > 460 ? 'text-danger' : 'text-success'}">${qtc > 460 ? 'Prolonged' : 'Normal'}</span></td>
            </tr>
            <tr>
                <td><strong>ST-Segment Deviation</strong></td>
                <td>${st > 0 ? '+' : ''}${st.toFixed(2)} mV</td>
                <td>-0.05 – +0.10 mV</td>
                <td><span class="${Math.abs(st) > 0.15 ? 'text-danger' : 'text-success'}">${st > 0.15 ? 'ST Elevation' : (st < -0.1 ? 'ST Depression' : 'Isoelectric')}</span></td>
            </tr>
            <tr>
                <td><strong>PR Interval</strong></td>
                <td>${pr.toFixed(1)} ms</td>
                <td>120 – 200 ms</td>
                <td><span class="${pr > 200 ? 'text-danger' : 'text-success'}">${pr > 200 ? '1st Degree AV Block' : 'Normal'}</span></td>
            </tr>
            <tr>
                <td><strong>SDNN (Autonomic HRV)</strong></td>
                <td>${sdnn.toFixed(1)} ms</td>
                <td>> 30.0 ms</td>
                <td><span class="${sdnn < 20 ? 'text-danger' : 'text-success'}">${sdnn < 20 ? 'Reduced HRV' : 'Intact'}</span></td>
            </tr>
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
                    biomarkers: (payload && payload.biomarkers) ? payload.biomarkers : {},
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
