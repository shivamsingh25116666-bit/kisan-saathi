# KISAN SAATHI 🌾
### Autonomous Multilingual AI Agricultural Services Assistant for Indian Farmers

> **Track:** Civic Pulse, Digital Governance & Public Information  
> **Core Mission:** Empower smallholder Indian farmers to speak or type naturally in their regional language, explain any agricultural problem or government-service requirement, and have an autonomous AI agent plan and execute the task as far as legally and technically possible.

---

## The Problem
A smallholder farmer in rural India loses **₹500 to ₹1,000 + multiple days of agricultural wages** just attempting to enroll in government schemes like **PMFBY Crop Insurance**, **PM-KISAN**, or **Kisan Credit Card (KCC)** — often facing illiterate navigation hurdles, server downtime, language barriers, and predatory agents.

## The Solution
**KISAN SAATHI** is a dual-channel (Mobile-First Web App + WhatsApp Bot) autonomous AI assistant:
1. **Multilingual Voice & Chat**: Speak or type in **Hindi, Garhwali, Punjabi, Marathi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia, Assamese, or English**.
2. **General-Purpose AI Planner**: Dynamic intent understanding, task decomposition, and tool composition. Not a rigid chatbot with pre-scripted questions.
3. **End-to-End PMFBY Crop Insurance**: From crop identification and premium calculation (1.5% - 2%) to document OCR, review gate, Playwright portal automation, legitimate OTP verification, and official/simulation receipt generation.
4. **Digital Document Vault**: Preprocessed under 95KB for government portals, OCR parsing with strict Aadhaar and bank account redaction.
5. **Agricultural Advisory Suite**: Mandi APMC prices vs MSP, PM-KISAN status, KCC 4% subsidized loan limits, Soil Health Card N-P-K fertilizer doses, crop damage 72-hour claims, and ICAR pest remedies.

---

## Architecture Overview

```
                      Farmer (Speech / Text / Photo)
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
      Web App (Kisan Saathi)                  WhatsApp (Twilio Gateway)
    Mobile-First SPA (Audio STT/TTS)         Webhook HMAC & Deduplication
               │                                         │
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │   FastAPI Gateway Router (/api/* & /webhook/twilio)        │
       │   • Language Detection (13 Indian languages & dialects)     │
       │   • Path-traversal safe media uploads (<95KB compression)   │
       │   • Fernet AES-CBC Session Encryption at rest               │
       └────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │   General-Purpose Task Planner & Tool Registry               │
       │   • Dynamic task decomposition & step planner               │
       │   • Verified official tools: PMFBY, PM-KISAN, KCC, Mandi,   │
       │     Soil Health, Pest Doctor, Damage Claims                 │
       │   • Separation of facts, assumptions & missing information  │
       └────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │   LangGraph Cyclic State Machine (app/agent/graph.py)       │
       │   [Node 1] Multimodal Vision OCR (Gemini 2.5 Flash)        │
       │   [Guardrail Gate] Deterministic Pydantic & Master Checksum │
       │   [Clarification Gate] Dialect Voice/Text Q to Farmer       │
       │   [Node 2] Geospatial Verification (STRtree Point-in-Poly)  │
       │   [Node 3] Self-Healing Playwright Browser Automation Loop  │
       │   [Node 4] Stateful OTP Checkpoint (Redis / Safe Fallback)  │
       │   [Node 3b] Browser Resumption & Confirmation Verification  │
       │   [Audit & Delivery] Watermarked/Official Receipt PDF       │
       └─────────────────────────────────────────────────────────────┘
```

---

## 7 Core Pillars of Implementation

| Pillar | Capability | Module | Verification Status |
|---|---|---|---|
| **1. Multilingual Voice & UX** | Web Speech API STT/TTS in 13 Indian languages (Hindi, Garhwali, Punjabi, Marathi, Bengali, etc.), mobile-first UI with green/earth palette, touch targets ≥48px. | `static/index.html`, `static/app.js`, `static/style.css` | **Tested & Verified** |
| **2. Dynamic Task Planner** | Intent understanding across informal speech, missing info identification, multi-step plan generation with approval checkpoints. | `app/agent/planner.py` | **24/24 Tests Passed** |
| **3. Extensible Tool Registry** | Verified official data: PMFBY (1.5-2% premium), PM-KISAN (₹6,000 DBT), KCC (4% subsidized loan), Mandi modal prices vs MSP, ICAR pest diagnostics. | `app/agent/tools.py` | **Tested & Verified** |
| **4. Document Vault & OCR** | OpenCV deskew & <95KB adaptive JPEG compression for government portals, Gemini 2.5 Flash structured OCR, Aadhaar masked to last 4 digits. | `app/senses/multimodal.py`, `app/api/kisan_saathi.py` | **Tested & Verified** |
| **5. PMFBY Execution Engine** | Self-healing Playwright automation, cascading LGD dropdown matching without blind guessing, legitimate OTP checkpoint. | `app/agent/browser_worker.py`, `app/agent/graph.py` | **29/29 Tests Passed** |
| **6. State & Session Security** | Fernet AES-CBC at-rest session encryption, Redis connection cooldown resilience, zero plaintext OTP logging, MessageSid deduplication. | `app/core/state_manager.py` | **15/15 Tests Passed** |
| **7. Production Deployment** | Dockerfile with Chromium, docker-compose with Redis, render.yaml, vercel.json. | `Dockerfile`, `docker-compose.yml`, `render.yaml` | **Ready for Deploy** |

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.11 or 3.12
- (Optional) Redis server (if not present, automatic encrypted in-memory fallback engages)

### 2. Local Setup
```bash
# Clone the repository
git clone https://github.com/your-username/kisan-agent.git
cd kisan-agent

# Create virtual environment
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your `GEMINI_API_KEY`. (All other variables have safe local defaults enabled).

### 4. Run Server
```bash
python main.py
```
Open your browser at **http://localhost:8000** to access the **Kisan Saathi** web interface.

---

## Running the Automated Test Suite

The test suite contains **150 automated tests** covering all modules:

```bash
# Run complete test suite
python -m pytest tests/ -v

# Run Kisan Saathi Web & Planner tests
python -m pytest tests/test_kisan_saathi.py -v

# Run StateManager & Redis tests
python -m pytest tests/test_state_manager.py -v

# Run Webhook & Deduplication tests
python -m pytest tests/test_webhook.py -v

# Run Browser Worker & LangGraph tests
python -m pytest tests/test_browser_worker.py tests/test_graph.py -v

# Run Guardrails, Spatial & Multimodal tests
python -m pytest tests/test_validators.py tests/test_spatial_router.py tests/test_multimodal.py -v
```

---

## Deployment Instructions

### Option 1: Docker & Docker Compose (Recommended)
```bash
# Build and run with Redis container
docker-compose up --build -d

# Check logs
docker-compose logs -f web
```

### Option 2: Render.com (Free-Tier Compatible)
1. Push repository to GitHub.
2. Link your repo on Render and select **Blueprint** using `render.yaml`.
3. Add your `GEMINI_API_KEY` in Render Dashboard Environment settings.
4. Render will automatically build the container, install chromium, and deploy.

### Option 3: Vercel Serverless
Deploy using `vercel.json`:
```bash
vercel --prod
```

---

## Critical Security, Privacy & Integrity Rules

1. **Aadhaar Privacy Directive**: Full 12-digit Aadhaar numbers are **NEVER** stored, logged, or returned in API responses. The regex firewall enforces `"[Aadhaar Redacted]"` across all state, logs, and database records. Only the last 4 digits (`aadhaar_last4`) are kept for farmer identification.
2. **Zero Plaintext OTP Logging**: OTP codes are processed in-memory under per-session locks and relayed to portal inputs. Digits are stripped from persistent state post-verification.
3. **Honest Status & Watermarking**: If operating in simulation mode due to portal maintenance, the generated PDF contains a bold, permanent watermark:
   `*** SIMULATION / NOT AN OFFICIAL GOVERNMENT RECEIPT ***`
   Simulated receipts are never passed off as genuine government issuances.
4. **Human-in-the-Loop Approval**: Irreversible actions (portal submission, payment) require explicit farmer confirmation before execution.
