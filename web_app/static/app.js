/**
 * ECG Foundation Representation System — Frontend Controller
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentSample = null;
    let currentSignal = null;
    let currentBiomarkers = null;
    let currentAnalysis = null;
    let currentLeads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
    let canvasMode = "12"; // "12" or "lead2"
    let radarChart = null;

    // DOM Elements
    const sampleSelect = document.getElementById("sampleSelect");
    const btnAnalyze = document.getElementById("btnAnalyze");
    const btnRegenLLM = document.getElementById("btnRegenLLM");
    const btnGrid12 = document.getElementById("btnGrid12");
    const btnGridLeadII = document.getElementById("btnGridLeadII");
    const ecgCanvas = document.getElementById("ecgCanvas");
    const saliencyCanvas = document.getElementById("saliencyCanvas");
    const spectrogramCanvas = document.getElementById("spectrogramCanvas");

    // Initialize App
    init();

    async function init() {
        setupEventListeners();
        await loadSamples();
    }

    function setupEventListeners() {
        sampleSelect.addEventListener("change", (e) => {
            loadSampleData(e.target.value);
        });

        btnAnalyze.addEventListener("click", () => {
            if (currentSignal) {
                runFullAnalysis();
            }
        });

        btnRegenLLM.addEventListener("click", () => {
            if (currentAnalysis) {
                generateLLMInterpretation();
            }
        });

        btnGrid12.addEventListener("click", () => {
            canvasMode = "12";
            btnGrid12.classList.add("active");
            btnGridLeadII.classList.remove("active");
            renderECGCanvas();
        });

        btnGridLeadII.addEventListener("click", () => {
            canvasMode = "lead2";
            btnGridLeadII.classList.add("active");
            btnGrid12.classList.remove("active");
            renderECGCanvas();
        });

        window.addEventListener("resize", () => {
            renderECGCanvas();
            if (currentAnalysis) {
                renderTemporalSaliency(currentAnalysis.visualizations.saliency_lead_ii);
                renderSpectrogram(currentAnalysis.visualizations.spectrogram_lead_ii);
            }
        });
    }

    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.status === "success" && data.samples.length > 0) {
                sampleSelect.innerHTML = "";
                data.samples.forEach(sample => {
                    const opt = document.createElement("option");
                    opt.value = sample.id;
                    opt.textContent = `[${sample.category}] ${sample.name}`;
                    sampleSelect.appendChild(opt);
                });
                // Load initial sample
                loadSampleData(data.samples[0].id);
            }
        } catch (err) {
            console.error("Failed to fetch samples:", err);
        }
    }

    async function loadSampleData(sampleId) {
        try {
            const res = await fetch(`/api/sample/${sampleId}`);
            const data = await res.json();
            if (data.status === "success") {
                currentSample = data.sample;
                currentSignal = data.signal;
                currentBiomarkers = data.biomarkers;
                currentLeads = data.leads;

                updatePatientProfile(currentSample, currentBiomarkers);
                renderECGCanvas();
                renderBiomarkers(currentBiomarkers);
                
                // Automatically run analysis on selection
                runFullAnalysis();
            }
        } catch (err) {
            console.error("Failed to load sample data:", err);
        }
    }

    function updatePatientProfile(sample, bio) {
        document.getElementById("valRecordId").textContent = sample.id;
        document.getElementById("valAgeSex").textContent = `${sample.age} / ${sample.sex}`;
        document.getElementById("valHeartRate").textContent = `${sample.heart_rate} bpm`;
        document.getElementById("valGroundTruth").textContent = sample.ground_truth.join(", ");
        document.getElementById("valHistory").textContent = sample.clinical_history;

        const badge = document.getElementById("patientCategoryBadge");
        badge.textContent = sample.category;
        if (sample.category === "NORM") {
            badge.className = "tag-status text-success";
        } else {
            badge.className = "tag-status text-danger";
        }
    }

    async function runFullAnalysis() {
        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = `<span>Analyzing...</span>`;

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    signal: currentSignal,
                    biomarkers: currentBiomarkers
                })
            });

            const data = await res.json();
            if (data.status === "success") {
                currentAnalysis = data;
                renderPredictions(data.predictions);
                renderTemporalSaliency(data.visualizations.saliency_lead_ii);
                renderSpectrogram(data.visualizations.spectrogram_lead_ii);
                renderTelemetry(data.representations);

                // Trigger Gemini LLM Interpretation
                generateLLMInterpretation();
            }
        } catch (err) {
            console.error("Analysis execution failed:", err);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                <span>Run Full Analysis</span>
            `;
        }
    }

    function renderPredictions(predictions) {
        const container = document.getElementById("probBarsContainer");
        container.innerHTML = "";

        const classLabels = {
            "NORM": "Normal Rhythm",
            "MI": "Myocardial Infarct",
            "STTC": "ST/T Abnormality",
            "CD": "Conduction Delay",
            "HYP": "Hypertrophy"
        };

        for (const [cname, prob] of Object.entries(predictions.probabilities)) {
            const thresh = predictions.thresholds[cname] || 0.5;
            const isDetected = predictions.binary_decisions[cname] === 1;
            const pct = Math.round(prob * 100);
            const threshPct = Math.round(thresh * 100);

            const card = document.createElement("div");
            card.className = `prob-metric-item ${isDetected ? 'detected' : ''}`;
            card.innerHTML = `
                <div class="prob-class-name">
                    <span>${cname}</span>
                    <span style="font-size:0.75rem; color: ${isDetected ? 'var(--rose)' : 'var(--emerald)'};">
                        ${isDetected ? 'POSITIVE' : 'NEGATIVE'}
                    </span>
                </div>
                <div class="prob-val-percent">${pct}%</div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                </div>
                <div class="prob-threshold-note">Thresh: ${threshPct}% • ${classLabels[cname] || ''}</div>
            `;
            container.appendChild(card);
        }
    }

    /* ─── 12-Lead ECG Canvas Renderer ─── */
    function renderECGCanvas() {
        if (!currentSignal) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = ecgCanvas.getBoundingClientRect();
        ecgCanvas.width = rect.width * dpr;
        ecgCanvas.height = rect.height * dpr;

        const ctx = ecgCanvas.getContext("2d");
        ctx.scale(dpr, dpr);
        const W = rect.width;
        const H = rect.height;

        // Draw Clinical Grid Background
        ctx.fillStyle = "#05070d";
        ctx.fillRect(0, 0, W, H);

        const gridSizeSmall = 8;
        const gridSizeMajor = 40;

        // Minor gridlines (1mm)
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = "rgba(244, 63, 94, 0.08)";
        for (let x = 0; x < W; x += gridSizeSmall) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, H);
            ctx.stroke();
        }
        for (let y = 0; y < H; y += gridSizeSmall) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        // Major gridlines (5mm)
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(244, 63, 94, 0.22)";
        for (let x = 0; x < W; x += gridSizeMajor) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, H);
            ctx.stroke();
        }
        for (let y = 0; y < H; y += gridSizeMajor) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(W, y);
            ctx.stroke();
        }

        if (canvasMode === "12") {
            // Standard 3x4 layout (3 rows, 4 columns)
            const rows = 3;
            const cols = 4;
            const cellW = W / cols;
            const cellH = H / rows;

            // Standard order: Column 1: I, II, III; Column 2: aVR, aVL, aVF; Column 3: V1, V2, V3; Column 4: V4, V5, V6
            const leadOrder = [
                ["I", "aVR", "V1", "V4"],
                ["II", "aVL", "V2", "V5"],
                ["III", "aVF", "V3", "V6"]
            ];

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const leadName = leadOrder[r][c];
                    const leadIdx = currentLeads.indexOf(leadName);
                    if (leadIdx === -1) continue;

                    const startX = c * cellW;
                    const startY = r * cellH;
                    const midY = startY + cellH / 2;

                    // Lead Label
                    ctx.font = "600 11px Inter, sans-serif";
                    ctx.fillStyle = "#06b6d4";
                    ctx.fillText(leadName, startX + 12, startY + 18);

                    // Plot Signal Segment
                    const rawLead = currentSignal[leadIdx]; // 1000 points
                    const pointsPerCol = Math.floor(rawLead.length / cols);
                    const slice = rawLead.slice(c * pointsPerCol, (c + 1) * pointsPerCol);

                    ctx.lineWidth = 1.4;
                    ctx.strokeStyle = "#00f2fe";
                    ctx.beginPath();

                    for (let i = 0; i < slice.length; i++) {
                        const px = startX + (i / (slice.length - 1)) * cellW;
                        const py = midY - slice[i] * 32; // 1mV = ~32px
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    }
                    ctx.stroke();
                }
            }
        } else {
            // Continuous Lead II Rhythm Strip
            const leadIdx = 1; // Lead II
            const rawLead = currentSignal[leadIdx];
            const midY = H / 2;

            ctx.font = "700 13px Inter, sans-serif";
            ctx.fillStyle = "#06b6d4";
            ctx.fillText("Lead II — Continuous Rhythm Strip (10 seconds @ 100 Hz)", 16, 26);

            ctx.lineWidth = 1.8;
            ctx.strokeStyle = "#00f2fe";
            ctx.beginPath();

            for (let i = 0; i < rawLead.length; i++) {
                const px = (i / (rawLead.length - 1)) * W;
                const py = midY - rawLead[i] * 45;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }
    }

    /* ─── Temporal Saliency Renderer ─── */
    function renderTemporalSaliency(saliency) {
        if (!saliency) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = saliencyCanvas.getBoundingClientRect();
        saliencyCanvas.width = rect.width * dpr;
        saliencyCanvas.height = rect.height * dpr;

        const ctx = saliencyCanvas.getContext("2d");
        ctx.scale(dpr, dpr);
        const W = rect.width;
        const H = rect.height;

        ctx.fillStyle = "#05070d";
        ctx.fillRect(0, 0, W, H);

        const leadSignal = currentSignal[1]; // Lead II
        const midY = H / 2;

        // Draw gradient attention heat map underneath
        for (let i = 0; i < saliency.length; i++) {
            const x = (i / (saliency.length - 1)) * W;
            const alpha = Math.min(1.0, saliency[i] * 1.5);
            ctx.fillStyle = `rgba(244, 63, 94, ${alpha * 0.4})`;
            ctx.fillRect(x, 0, W / saliency.length + 1, H);
        }

        // Draw Waveform on top
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = "#ffffff";
        ctx.beginPath();
        for (let i = 0; i < leadSignal.length; i++) {
            const px = (i / (leadSignal.length - 1)) * W;
            const py = midY - leadSignal[i] * 28;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    /* ─── Spectrogram Heatmap Renderer ─── */
    function renderSpectrogram(spec) {
        if (!spec) return;
        const dpr = window.devicePixelRatio || 1;
        const rect = spectrogramCanvas.getBoundingClientRect();
        spectrogramCanvas.width = rect.width * dpr;
        spectrogramCanvas.height = rect.height * dpr;

        const ctx = spectrogramCanvas.getContext("2d");
        ctx.scale(dpr, dpr);
        const W = rect.width;
        const H = rect.height;

        const freqBins = spec.length;      // 33
        const timeSteps = spec[0].length;  // ~32
        const cellW = W / timeSteps;
        const cellH = H / freqBins;

        for (let f = 0; f < freqBins; f++) {
            for (let t = 0; t < timeSteps; t++) {
                const val = spec[f][t];
                // Viridis/Plasma pseudo-colormap
                const norm = Math.min(1.0, val / 4.0);
                const r = Math.floor(255 * norm);
                const g = Math.floor(180 * (1 - Math.abs(norm - 0.5) * 2));
                const b = Math.floor(255 * (1 - norm));
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(t * cellW, H - (f + 1) * cellH, cellW + 0.5, cellH + 0.5);
            }
        }
    }

    /* ─── Biomarker Radar & Table ─── */
    function renderBiomarkers(bio) {
        const radarLabels = ["HRV SDNN", "QRS Duration", "QTc Bazett", "ST Deviation", "R Amplitude", "PR Interval"];
        const patientVals = [
            Math.min(100, (bio.SDNN / 50) * 50),
            Math.min(100, (bio.QRS_Duration / 120) * 50),
            Math.min(100, (bio.QTc_Bazett / 450) * 50),
            Math.min(100, 50 + bio.ST_Deviation * 40),
            Math.min(100, (bio.R_Amplitude / 1.5) * 50),
            Math.min(100, (bio.PR_Interval / 200) * 50),
        ];

        const normalVals = [50, 50, 50, 50, 50, 50];

        const ctx = document.getElementById("biomarkerRadarChart").getContext("2d");
        if (radarChart) radarChart.destroy();

        radarChart = new Chart(ctx, {
            type: "radar",
            data: {
                labels: radarLabels,
                datasets: [
                    {
                        label: "Patient ECG",
                        data: patientVals,
                        backgroundColor: "rgba(6, 182, 212, 0.25)",
                        borderColor: "#06b6d4",
                        pointBackgroundColor: "#06b6d4",
                        borderWidth: 2
                    },
                    {
                        label: "Standard Reference",
                        data: normalVals,
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
                    legend: {
                        labels: { color: "#9ca3af", font: { family: "Inter", size: 11 } }
                    }
                }
            }
        });

        // Table Rows
        const tbody = document.getElementById("biomarkerTableBody");
        tbody.innerHTML = `
            <tr>
                <td><strong>QRS Duration</strong></td>
                <td>${bio.QRS_Duration.toFixed(1)} ms</td>
                <td>70 – 110 ms</td>
                <td><span class="${bio.QRS_Duration > 120 ? 'text-danger' : 'text-success'}">${bio.QRS_Duration > 120 ? 'Prolonged (Conduction Delay)' : 'Normal'}</span></td>
            </tr>
            <tr>
                <td><strong>QTc (Bazett)</strong></td>
                <td>${bio.QTc_Bazett.toFixed(1)} ms</td>
                <td>360 – 440 ms</td>
                <td><span class="${bio.QTc_Bazett > 460 ? 'text-danger' : 'text-success'}">${bio.QTc_Bazett > 460 ? 'Borderline / Prolonged' : 'Normal'}</span></td>
            </tr>
            <tr>
                <td><strong>ST-Segment Deviation</strong></td>
                <td>${bio.ST_Deviation > 0 ? '+' : ''}${bio.ST_Deviation.toFixed(2)} mV</td>
                <td>-0.05 – +0.10 mV</td>
                <td><span class="${Math.abs(bio.ST_Deviation) > 0.15 ? 'text-danger' : 'text-success'}">${bio.ST_Deviation > 0.15 ? 'ST Elevation' : (bio.ST_Deviation < -0.1 ? 'ST Depression' : 'Isoelectric')}</span></td>
            </tr>
            <tr>
                <td><strong>PR Interval</strong></td>
                <td>${bio.PR_Interval.toFixed(1)} ms</td>
                <td>120 – 200 ms</td>
                <td><span class="${bio.PR_Interval > 200 ? 'text-danger' : 'text-success'}">${bio.PR_Interval > 200 ? '1st Degree AV Block' : 'Normal'}</span></td>
            </tr>
            <tr>
                <td><strong>SDNN (HRV)</strong></td>
                <td>${bio.SDNN.toFixed(1)} ms</td>
                <td>> 30.0 ms</td>
                <td><span class="${bio.SDNN < 20 ? 'text-danger' : 'text-success'}">${bio.SDNN < 20 ? 'Reduced Autonomic Tone' : 'Intact HRV'}</span></td>
            </tr>
        `;
    }

    function renderTelemetry(rep) {
        document.getElementById("codeTemporal").textContent = `[${rep.z_temporal_sample.slice(0, 8).map(v => v.toFixed(3)).join(", ")} ... (${rep.z_temporal_dim}D)]`;
        document.getElementById("codeMorphology").textContent = `[${rep.z_morphology_sample.slice(0, 8).map(v => v.toFixed(3)).join(", ")} ... (${rep.z_morphology_dim}D)]`;
    }

    /* ─── Gemini LLM Report Generator ─── */
    async function generateLLMInterpretation() {
        const container = document.getElementById("llmReportContent");
        container.innerHTML = `
            <div class="llm-loading-state">
                <span class="spinner"></span>
                <p>Generating expert cardiological synthesis from multimodal representations via Gemini API...</p>
            </div>
        `;

        try {
            const res = await fetch("/api/interpret", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    record_id: currentSample.id,
                    detected_conditions: currentAnalysis.predictions.detected_conditions,
                    probabilities: currentAnalysis.predictions.probabilities,
                    thresholds: currentAnalysis.predictions.thresholds,
                    biomarkers: currentBiomarkers,
                    patient_metadata: {
                        Age: currentSample.age,
                        Sex: currentSample.sex,
                        History: currentSample.clinical_history
                    }
                })
            });

            const data = await res.json();
            if (data.report_markdown) {
                container.innerHTML = marked.parse(data.report_markdown);
            } else {
                container.innerHTML = `<p class="text-danger">Failed to generate LLM consultation: ${data.message || 'Unknown error'}</p>`;
            }
        } catch (err) {
            console.error("LLM interpretation failed:", err);
            container.innerHTML = `<p class="text-danger">Error connecting to Gemini clinical interpreter service.</p>`;
        }
    }
});
