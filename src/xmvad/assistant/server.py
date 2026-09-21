"""Lightweight HTTP server providing API endpoints and Dashboard UI for PNTC AI Assistant.

Phase A15 (API Backend), Phase A23 (UI with ASK PNTC panel), Phase A24 (Provider Switch),
Phase A25 (Faculty Demo Mode), and Phase A26 (Viva Mode).
"""

from __future__ import annotations

import json
import logging
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .chat import PNTCAssistant
from .provider_factory import create_provider, get_provider_status
from .schema import ResponseMode

logger = logging.getLogger("pntc.assistant.server")

# Canonical sample repository for demonstration and testing
CANONICAL_SAMPLES: Dict[str, Dict[str, Any]] = {
    "sample_001": {
        "sample": {
            "id": "sample_001",
            "category": "potato",
        },
        "inspection": {
            "decision": "anomalous",
            "anomaly_score": 0.91,
            "inspection_status": "DEFECT_DETECTED",
            "reliability": 0.89,
            "manual_review_recommended": False,
        },
        "defects": [
            {
                "id": 1,
                "location": {
                    "label": "upper-right",
                    "centroid_normalized": [0.74, 0.21],
                },
                "shape": {
                    "label": "elongated irregular",
                    "aspect_ratio": 2.91,
                    "orientation_deg": 31.2,
                },
                "size": {
                    "major_length_mm": 18.4,
                    "minor_length_mm": 9.7,
                    "surface_area_mm2": 132.4,
                },
                "geometry": {
                    "structure": "localized depression",
                    "max_depression_mm": 1.82,
                    "mean_depression_mm": 0.71,
                },
                "volume": {
                    "missing_material_mm3": 84.6,
                    "excess_material_mm3": 2.4,
                    "measurement_confidence": 0.89,
                },
                "evidence": {
                    "rgb": {
                        "value": 0.61,
                        "level": "Moderate",
                    },
                    "xyz": {
                        "value": 0.83,
                        "level": "High",
                    },
                    "topology_disagreement": {
                        "value": 0.91,
                        "level": "Very High",
                    },
                },
                "normal_reference": {
                    "prototype_id": 1842,
                    "training_sample_id": "normal_train_042",
                },
                "prototype_trace": {
                    "shared_top5": 1,
                    "rgb_top5": [1842, 1021, 934, 401, 88],
                    "xyz_top5": [1842, 2301, 1400, 781, 1920],
                    "jaccard": 0.111,
                    "js_divergence": 0.87,
                    "gate": 0.91,
                },
                "quality": {
                    "xyz_valid_fraction": 0.94,
                    "geometry_measurement_confidence": 0.89,
                },
            }
        ],
    },
    "sample_normal_001": {
        "sample": {
            "id": "sample_normal_001",
            "category": "potato",
        },
        "inspection": {
            "decision": "normal",
            "anomaly_score": 0.12,
            "inspection_status": "NORMAL",
            "reliability": 0.98,
            "manual_review_recommended": False,
        },
        "defects": [],
    },
}

# Global assistant instance
_assistant_instance: Optional[PNTCAssistant] = None


def get_assistant() -> PNTCAssistant:
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = PNTCAssistant()
    return _assistant_instance


def set_assistant_provider(provider_name: str) -> Dict[str, Any]:
    global _assistant_instance
    prov = create_provider(provider_name)
    _assistant_instance = PNTCAssistant(provider=prov)
    return get_provider_status(prov)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PNTC Industrial Inspection Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f19;
      --bg-card: #131b2e;
      --bg-surface: #1c2742;
      --accent: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.25);
      --accent-green: #10b981;
      --accent-orange: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #94a3b8;
      --border: #233152;
      --border-light: #2d3e66;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }
    header {
      background: var(--bg-card);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .badge-frozen {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.72rem;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 3px 8px;
      border-radius: 999px;
      font-weight: 600;
    }
    .header-controls {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .provider-select {
      background: var(--bg-surface);
      color: var(--text-main);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 500;
      outline: none;
      cursor: pointer;
    }
    .mode-select {
      background: var(--bg-surface);
      color: var(--text-main);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
    }
    main {
      display: flex;
      flex: 1;
      height: calc(100vh - 65px);
      overflow: hidden;
    }
    /* Left Panel - Inspection Dashboard */
    .left-panel {
      flex: 1.2;
      padding: 24px;
      overflow-y: auto;
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .card-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
    }
    .stat-item {
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
    }
    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 4px;
    }
    .stat-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.1rem;
      font-weight: 600;
    }
    .val-anom { color: var(--accent-red); }
    .val-norm { color: var(--accent-green); }
    .val-high { color: var(--accent-orange); }
    .val-accent { color: #60a5fa; }

    /* Right Panel - ASK PNTC */
    .right-panel {
      flex: 1;
      background: var(--bg-card);
      display: flex;
      flex-direction: column;
      height: 100%;
    }
    .chat-header {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(19, 27, 46, 0.85);
      backdrop-filter: blur(8px);
    }
    .chat-title {
      font-size: 1.05rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .provider-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.75rem;
      background: rgba(59, 130, 246, 0.15);
      color: #93c5fd;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 3px 10px;
      border-radius: 999px;
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      background-color: #3b82f6;
      border-radius: 50%;
      box-shadow: 0 0 8px #3b82f6;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.8; }
      50% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.8; }
    }
    .chat-body {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .message-bubble {
      max-width: 85%;
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 0.92rem;
      line-height: 1.55;
    }
    .user-msg {
      align-self: flex-end;
      background: #2563eb;
      color: #ffffff;
      border-bottom-right-radius: 2px;
    }
    .assistant-msg {
      align-self: flex-start;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      color: var(--text-main);
      border-bottom-left-radius: 2px;
    }
    .msg-meta {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 6px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }
    .ui-action-chip {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: rgba(16, 185, 129, 0.2);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #6ee7b7;
      font-size: 0.75rem;
      padding: 3px 8px;
      border-radius: 6px;
      margin-top: 6px;
      font-family: 'JetBrains Mono', monospace;
    }
    .quick-bar {
      padding: 10px 16px;
      border-top: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      background: rgba(11, 15, 25, 0.6);
    }
    .quick-btn {
      background: var(--bg-surface);
      color: #94a3b8;
      border: 1px solid var(--border);
      padding: 5px 10px;
      border-radius: 6px;
      font-size: 0.78rem;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .quick-btn:hover {
      background: var(--border);
      color: #ffffff;
      border-color: #3b82f6;
    }
    .chat-footer {
      padding: 14px 16px;
      border-top: 1px solid var(--border);
      background: var(--bg-card);
      display: flex;
      gap: 10px;
    }
    .chat-input {
      flex: 1;
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      color: var(--text-main);
      font-size: 0.9rem;
      outline: none;
    }
    .chat-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent-glow);
    }
    .send-btn {
      background: var(--accent);
      color: #ffffff;
      border: none;
      padding: 0 18px;
      border-radius: 8px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .send-btn:hover { background: #2563eb; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-title">PNTC Industrial Inspection</span>
      <span class="badge-frozen">CANONICAL I-AUROC 0.96541</span>
      <span class="badge-frozen">h5d-pntc-verified</span>
    </div>
    <div class="header-controls">
      <select id="sampleSelect" class="mode-select" onchange="loadSample()">
        <option value="sample_001">sample_001 (Potato Anomaly)</option>
        <option value="sample_normal_001">sample_normal_001 (Potato Normal)</option>
      </select>
      <select id="modeSelect" class="mode-select">
        <option value="TECHNICAL">Mode: Technical</option>
        <option value="SIMPLE">Mode: Simple</option>
        <option value="VIVA">Mode: Viva / Defense</option>
      </select>
      <select id="providerSelect" class="provider-select" onchange="switchProvider()">
        <option value="gemini">Gemini API</option>
        <option value="grok">xAI Grok API</option>
      </select>
    </div>
  </header>

  <main>
    <div class="left-panel">
      <div class="card">
        <div class="card-title">
          <span>Inspection Status</span>
          <span id="decisionBadge" style="font-family:'JetBrains Mono';font-size:0.85rem;color:#ef4444;">DEFECT DETECTED</span>
        </div>
        <div class="stat-grid">
          <div class="stat-item">
            <div class="stat-label">Decision</div>
            <div class="stat-value val-anom" id="valDecision">ANOMALOUS</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Anomaly Score</div>
            <div class="stat-value" id="valScore">0.91</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Reliability</div>
            <div class="stat-value val-accent" id="valReliability">0.89</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Category</div>
            <div class="stat-value" id="valCategory">potato</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">3D Defect Measurements (Defect #1)</div>
        <div class="stat-grid">
          <div class="stat-item">
            <div class="stat-label">Max Depression</div>
            <div class="stat-value val-accent" id="valDepression">1.82 mm</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Missing Material</div>
            <div class="stat-value val-high" id="valMissing">84.6 mm³</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Surface Area</div>
            <div class="stat-value" id="valArea">132.4 mm²</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Morphology</div>
            <div class="stat-value" id="valShape" style="font-size:0.95rem;">elongated irreg.</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">PNTC Multimodal Topology & Retrieval</div>
        <div class="stat-grid">
          <div class="stat-item">
            <div class="stat-label">RGB Evidence</div>
            <div class="stat-value">0.61 (Moderate)</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">XYZ Evidence</div>
            <div class="stat-value val-high">0.83 (High)</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Topology Disagreement</div>
            <div class="stat-value val-anom">0.91 (Very High)</div>
          </div>
          <div class="stat-item">
            <div class="stat-label">Paired Prototype Twin</div>
            <div class="stat-value val-accent">#1842</div>
          </div>
        </div>
      </div>
    </div>

    <div class="right-panel">
      <div class="chat-header">
        <div class="chat-title">
          <span>ASK PNTC</span>
        </div>
        <div class="provider-pill">
          <div class="pulse-dot"></div>
          <span id="providerDisplay">Gemini 2.5 Flash</span>
        </div>
      </div>

      <div class="chat-body" id="chatContainer">
        <!-- Messages rendered dynamically -->
      </div>

      <div class="quick-bar">
        <button class="quick-btn" onclick="sendQuickPrompt('Explain defect')">Explain defect</button>
        <button class="quick-btn" onclick="sendQuickPrompt('Explain PNTC')">Explain PNTC</button>
        <button class="quick-btn" onclick="sendQuickPrompt('Show me the nearest normal example.')">Show normal reference</button>
        <button class="quick-btn" onclick="sendQuickPrompt('Show prototype evidence')">Show prototype evidence</button>
        <button class="quick-btn" onclick="sendQuickPrompt('Explain geometry')">Explain geometry</button>
        <button class="quick-btn" onclick="sendQuickPrompt('How reliable is the geometry measurement?')">Measurement reliability</button>
      </div>

      <div class="chat-footer">
        <input type="text" id="userInput" class="chat-input" placeholder="Ask about this inspection or viva questions..." onkeypress="handleKey(event)">
        <button class="send-btn" onclick="sendMessage()">Send</button>
      </div>
    </div>
  </main>

  <script>
    let conversationId = 'conv_' + Math.random().toString(36).substring(2, 9);
    let currentSampleId = 'sample_001';

    function addMessage(role, text, meta, uiAction) {
      const c = document.getElementById('chatContainer');
      const div = document.createElement('div');
      div.className = 'message-bubble ' + (role === 'user' ? 'user-msg' : 'assistant-msg');
      let html = '<div>' + escapeHtml(text) + '</div>';
      if (uiAction) {
        html += '<div class="ui-action-chip">⚡ ACTION: ' + uiAction.type + (uiAction.prototype_id ? ' (#' + uiAction.prototype_id + ')' : '') + '</div>';
      }
      if (meta) {
        html += '<div class="msg-meta"><span>' + meta.model + '</span><span>' + (meta.latency_ms ? meta.latency_ms + ' ms' : 'Grounded') + '</span></div>';
      }
      div.innerHTML = html;
      c.appendChild(div);
      c.scrollTop = c.scrollHeight;
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    async function loadSample() {
      currentSampleId = document.getElementById('sampleSelect').value;
      const mode = document.getElementById('modeSelect').value;
      const res = await fetch('/api/assistant/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_id: currentSampleId, response_mode: mode })
      });
      const data = await res.json();
      document.getElementById('chatContainer').innerHTML = '';
      addMessage('assistant', data.message, { model: data.model, latency_ms: data.latency_ms });

      if (currentSampleId === 'sample_normal_001') {
        document.getElementById('decisionBadge').innerText = 'NORMAL';
        document.getElementById('decisionBadge').style.color = '#10b981';
        document.getElementById('valDecision').innerText = 'NORMAL';
        document.getElementById('valDecision').className = 'stat-value val-norm';
        document.getElementById('valScore').innerText = '0.12';
        document.getElementById('valDepression').innerText = '0.00 mm';
        document.getElementById('valMissing').innerText = '0.0 mm³';
      } else {
        document.getElementById('decisionBadge').innerText = 'DEFECT DETECTED';
        document.getElementById('decisionBadge').style.color = '#ef4444';
        document.getElementById('valDecision').innerText = 'ANOMALOUS';
        document.getElementById('valDecision').className = 'stat-value val-anom';
        document.getElementById('valScore').innerText = '0.91';
        document.getElementById('valDepression').innerText = '1.82 mm';
        document.getElementById('valMissing').innerText = '84.6 mm³';
      }
    }

    async function sendMessage() {
      const input = document.getElementById('userInput');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      addMessage('user', text);
      const mode = document.getElementById('modeSelect').value;

      try {
        const res = await fetch('/api/assistant/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_id: currentSampleId,
            message: text,
            conversation_id: conversationId,
            response_mode: mode
          })
        });
        const data = await res.json();
        addMessage('assistant', data.message, { model: data.model, latency_ms: data.latency_ms }, data.ui_action);
      } catch (err) {
        addMessage('assistant', 'Error communicating with assistant backend.');
      }
    }

    function sendQuickPrompt(prompt) {
      document.getElementById('userInput').value = prompt;
      sendMessage();
    }

    function handleKey(e) {
      if (e.key === 'Enter') sendMessage();
    }

    async function switchProvider() {
      const p = document.getElementById('providerSelect').value;
      const res = await fetch('/api/assistant/provider', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: p })
      });
      const data = await res.json();
      document.getElementById('providerDisplay').innerText = data.provider === 'gemini' ? 'Gemini 2.5 Flash' : 'Grok 2';
    }

    // Initial load
    window.onload = () => {
      loadSample();
    };
  </script>
</body>
</html>
"""


class AssistantHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for PNTC Assistant API."""

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/dashboard"):
            raw = DASHBOARD_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        if path == "/api/assistant/provider":
            asst = get_assistant()
            status = get_provider_status(asst.provider)
            self._send_json(status)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(length) if length > 0 else b"{}"

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Malformed JSON")
            return

        if path == "/api/assistant/provider":
            prov_name = body.get("provider", "gemini")
            status = set_assistant_provider(prov_name)
            self._send_json(status)
            return

        asst = get_assistant()
        sample_id = body.get("sample_id", "sample_001")
        sample_data = CANONICAL_SAMPLES.get(sample_id, CANONICAL_SAMPLES["sample_001"])
        mode_str = body.get("response_mode", "TECHNICAL").upper()
        try:
            mode = ResponseMode(mode_str)
        except ValueError:
            mode = ResponseMode.TECHNICAL

        if path == "/api/assistant/explain":
            force_refresh = bool(body.get("force_refresh", False))
            resp = asst.generate_inspection_summary(
                inspection_result=sample_data,
                response_mode=mode,
                force_refresh=force_refresh,
            )
            self._send_json(resp.to_dict())
            return

        if path == "/api/assistant/chat":
            msg = body.get("message", "")
            cid = body.get("conversation_id")
            did = body.get("defect_id")
            resp = asst.chat(
                sample_id=sample_id,
                message=msg,
                conversation_id=cid,
                inspection_result=sample_data,
                defect_id=did,
                response_mode=mode,
            )
            self._send_json(resp.to_dict())
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def log_message(self, format: str, *args: Any) -> None:
        # Safe logging: never prints body or headers containing credentials
        logger.info("%s - - [%s] %s", self.client_address[0], self.log_date_time_string(), format % args)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the PNTC Assistant HTTP server."""
    server_addr = (host, port)
    httpd = ThreadingHTTPServer(server_addr, AssistantHTTPHandler)
    logger.info("Starting PNTC AI Assistant server at http://%s:%d", host, port)
    print(f"PNTC AI Assistant server running at http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping PNTC AI Assistant server...")
        httpd.server_close()
