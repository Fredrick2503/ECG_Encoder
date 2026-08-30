"""
ECG Foundation Representation Web Application Server
===================================================
Standalone web server hosting the multimodal ECG cockpit and calling ECGEncoderEngine.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import mimetypes
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
import torch

# Add project root to sys.path so engine can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecg_engine import ECGEncoderEngine, EngineConfig
from morphology_encoder.conversion import ecg_to_spectrogram
from web_app.sample_manager import SampleManager, LEAD_NAMES
from web_app.llm_service import GeminiClinicalInterpreter

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Global Singletons
sample_manager = SampleManager()
llm_interpreter = GeminiClinicalInterpreter()
engine: Optional[ECGEncoderEngine] = None


def get_engine() -> ECGEncoderEngine:
    global engine
    if engine is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = EngineConfig(device=device)
        engine = ECGEncoderEngine(config=cfg)
    return engine


class ECGWebAppHandler(BaseHTTPRequestHandler):
    
    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        # API Routes
        if path == "/api/embeddings_3d" or path == "/api/embeddings":
            emb = sample_manager.get_3d_embeddings()
            self._send_json(emb)
            return

        elif path == "/api/records" or path == "/api/samples":
            params = parse_qs(parsed.query)
            try:
                page = max(1, int(params.get("page", ["1"])[0]))
            except ValueError:
                page = 1
            try:
                limit = max(1, min(100, int(params.get("limit", ["20"])[0])))
            except ValueError:
                limit = 20

            cat = params.get("category", ["ALL"])[0].strip()
            search = params.get("search", [""])[0].lower().strip()

            all_records = sample_manager.get_all_records()
            filtered = []
            for r in all_records:
                if cat != "ALL" and r.get("category") != cat:
                    continue
                if search:
                    text_blob = f"{r.get('id', '')} {r.get('sample_code', '')} {r.get('name', '')} {r.get('category', '')} {r.get('clinical_history', '')} {r.get('age', '')}".lower()
                    if search not in text_blob:
                        continue
                filtered.append(r)

            total_count = len(filtered)
            start_idx = (page - 1) * limit
            end_idx = min(start_idx + limit, total_count)
            paginated_records = filtered[start_idx:end_idx]

            self._send_json({
                "status": "success",
                "total_count": total_count,
                "page": page,
                "limit": limit,
                "total_pages": max(1, (total_count + limit - 1) // limit),
                "records": paginated_records,
                "samples": paginated_records
            })
            return
            
        elif path.startswith("/api/sample/"):
            sample_id = path.replace("/api/sample/", "").strip()
            payload = sample_manager.generate_full_sample_payload(sample_id)
            self._send_json({
                "status": "success",
                "payload": payload,
                "sample": payload.get("record", {}),
                "leads": payload.get("leads", LEAD_NAMES),
                "signal": payload.get("signal", []),
                "biomarkers": payload.get("biomarkers", {})
            })
            return

        # Static File Serving
        if path == "/" or path == "":
            file_path = STATIC_DIR / "index.html"
        else:
            rel_path = path.lstrip("/")
            file_path = STATIC_DIR / rel_path

        if file_path.exists() and file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            req_data = json.loads(post_body.decode("utf-8"))
        except Exception:
            req_data = {}

        if path == "/api/analyze":
            try:
                eng = get_engine()
                signal_data = req_data.get("signal")
                raw_bio = req_data.get("biomarkers")
                
                if signal_data is None:
                    self._send_json({"status": "error", "message": "Missing signal data"}, 400)
                    return
                    
                signal_arr = np.array(signal_data, dtype=np.float32) # (12, 1000)
                
                # Execute Engine Pipeline
                rep, pred = eng.process(signal_arr)
                
                # Generate Spectrogram Representation for Morphology visualization
                with torch.no_grad():
                    x_t = eng.preprocessor.preprocess(signal_arr).to(eng.device)
                    spec_tensor = ecg_to_spectrogram(x_t).cpu().numpy()[0] # (12, freq, time)
                    
                    # Compute temporal saliency gradient proxy (activation energy across leads)
                    saliency_map = np.abs(np.diff(signal_arr, axis=1, prepend=signal_arr[:, :1]))
                    # Normalize saliency to [0, 1] per lead
                    s_min = saliency_map.min(axis=1, keepdims=True)
                    s_max = saliency_map.max(axis=1, keepdims=True) + 1e-6
                    saliency_norm = (saliency_map - s_min) / (s_max - s_min)

                response_data = {
                    "status": "success",
                    "representations": {
                        "z_temporal_dim": rep.z_temporal.shape[1],
                        "z_morphology_dim": rep.z_morphology.shape[1],
                        "z_biomarker_dim": rep.z_biomarker.shape[1],
                        "z_fused_dim": rep.z_fused.shape[1],
                        "z_temporal_sample": rep.z_temporal[0, :32].tolist(),
                        "z_morphology_sample": rep.z_morphology[0, :32].tolist(),
                        "z_biomarker_sample": rep.z_biomarker[0, :32].tolist(),
                        "z_fused_sample": rep.z_fused[0, :64].tolist(),
                    },
                    "predictions": {
                        "probabilities": {
                            cname: float(p) for cname, p in zip(pred.class_names, pred.probabilities[0])
                        },
                        "thresholds": {
                            cname: float(th) for cname, th in zip(pred.class_names, pred.thresholds)
                        },
                        "binary_decisions": {
                            cname: int(d) for cname, d in zip(pred.class_names, pred.predictions[0])
                        },
                        "detected_conditions": pred.detected_conditions[0]
                    },
                    "visualizations": {
                        "spectrogram_lead_ii": spec_tensor[1].tolist(), # Lead II Spectrogram
                        "saliency_lead_ii": saliency_norm[1].tolist(),
                    }
                }
                self._send_json(response_data)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"status": "error", "message": str(e)}, 500)
            return

        elif path == "/api/interpret":
            try:
                record_id = req_data.get("record_id", "SAMPLE-01")
                detected = req_data.get("detected_conditions", ["Normal Sinus Rhythm"])
                probs = req_data.get("probabilities", {})
                thresholds = req_data.get("thresholds", {})
                biomarkers = req_data.get("biomarkers", {})
                patient_meta = req_data.get("patient_metadata", {})
                
                report = llm_interpreter.generate_clinical_report(
                    record_id=record_id,
                    detected_conditions=detected,
                    probabilities=probs,
                    thresholds=thresholds,
                    biomarkers=biomarkers,
                    patient_metadata=patient_meta
                )
                self._send_json(report)
            except Exception as e:
                self._send_json({"status": "error", "message": str(e)}, 500)
            return

        self.send_error(404, "Endpoint not found")


def start_server(port: int = 8080):
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, ECGWebAppHandler)
    print("=" * 70)
    print(f"[*] ECG Foundation System Web Interface Running at: http://localhost:{port}")
    print(f"[*] Serving static UI from: {STATIC_DIR}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start ECG Multimodal Web Application")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind server (default: 8080)")
    args = parser.parse_args()
    start_server(port=args.port)
