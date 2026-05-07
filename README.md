# Chloe — Personal Holographic AI Assistant

> A fully local, multimodal AI assistant with a holographic interface,
> real-time voice pipeline, persistent memory, and live Bitcoin Lightning
> integration. Built end-to-end by Edward Wayne.

![Demo GIF placeholder — replace with demo.gif when recorded]

---

## What Is Chloe?

Chloe is an end-to-end personal AI assistant built for immersive, real-world use.
She runs locally on Windows, listens for a wake word, understands voice and vision,
remembers context across conversations, and can send and receive Bitcoin Lightning
payments — all through a holographic heads-up display rendered in a native desktop window.

This is not a wrapper around a chatbot API. It is a full real-time system: audio
pipeline, state machine, multimodal LLM routing, financial API integration,
persistent memory, and a custom 3D holographic UI — designed and built from scratch.

---

## Live Demo

> 📹 Demo video coming soon — wake word trigger, voice interaction, orb state
> transitions, and Lightning payment flow.

---

## Core Features

### 🎙️ Voice Pipeline
- Wake word detection via OpenWakeWord ("Hey Chloe")
- Speech-to-text via Groq Whisper (`whisper-large-v3-turbo`)
- LLM inference via Groq (`llama-3.3-70b-versatile`)
- Text-to-speech via ElevenLabs neural voice
- Full state machine: **Idle → Listening → Thinking → Speaking**
- Both voice and text chat paths share a unified conversation history

### 🌐 Holographic Interface
- Galaxy orb visualizer built in **Three.js** with custom **GLSL shaders**
- Real-time audio amplitude reactivity — orb and avatar face respond to voice
- Post-processing pipeline: Unreal Bloom, chromatic aberration, scanlines, vignette
- Canvas-rendered animated AI avatar face with lip sync
- WebSocket bridge connecting Python backend to browser-based HUD
- Packaged in a native desktop window via **PyQt6 + QWebEngineView**

### 👁️ Multimodal Understanding
- Image and screenshot analysis
- Video file input
- URL and link content ingestion
- Automatic model routing — switches to **Llama 4 Scout** (vision) when
  visual content is detected, text model otherwise

### ⚡ Bitcoin & Lightning Integration
- Send and receive Lightning invoices via **Breeze SDK**
- Live wallet balance displayed in the HUD
- Voice-triggered payment flows — fully hands-free

### 🧠 Persistent Memory
- Conversation history shared across voice and text input paths
- Long-term memory that grows over time, retaining important context
- User-designated memory — tell Chloe what to remember permanently
- History trimming keeps token costs low while preserving meaningful context

### 🔍 Web Search
- Real-time search via Groq's `compound-mini` model — server-side tool calls handle the search loop end-to-end
- A small router heuristic detects time-sensitive queries and routes only those to the search-capable model, saving quota for what only it can do
- Enables Chloe to answer questions beyond her training data cutoff

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        USER                              │
│          Voice / Text / Image / Video / URL              │
└─────────────────────┬────────────────────────────────────┘
                      │
          ┌───────────▼────────────┐
          │    HUD  (hud.html)     │  ← Three.js orb, canvas avatar face,
          │    Browser Frontend    │    chat interface, state display,
          │    (PyQt6 window)      │    attachment handling
          └───────────┬────────────┘
                      │  WebSocket  ws://localhost:6789
          ┌───────────▼────────────┐
          │    hud_server.py       │  ← Async WebSocket bridge
          │                        │    Broadcasts state to all clients
          └───────────┬────────────┘
                      │
          ┌───────────▼────────────┐
          │      jarvis.py         │  ← Core brain
          │  ┌─ Voice loop         │    Wake word → record → transcribe
          │  ├─ Chat handler       │    Text/vision → stream response
          │  ├─ Memory system      │    Persistent cross-modal context
          │  └─ Bitcoin/Lightning  │    Breeze SDK send/receive/balance
          └────────────────────────┘
                      │
      ┌───────────────┼────────────────┬────────────────┐
      │               │                │                │
  Groq API       ElevenLabs        Breeze SDK      Tavily API
  LLM + STT      Neural TTS        Lightning       Web Search
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — Llama 3.3 70B (text) + Llama 4 Scout 17B (vision) |
| STT | Groq Whisper large-v3-turbo |
| TTS | ElevenLabs neural voice |
| Wake Word | OpenWakeWord (ONNX inference) |
| 3D Visuals | Three.js r160 — custom GLSL vertex + fragment shaders |
| Post-FX | UnrealBloomPass, chromatic aberration, scanline pass |
| Avatar | HTML5 Canvas — procedural face with lip sync |
| Backend | Python — asyncio, websockets, threading |
| Desktop | PyQt6 + QWebEngineView |
| Bitcoin | Breeze SDK — Lightning Network send/receive/balance |
| Search | Tavily API |
| Packaging | PyInstaller — standalone .exe, no Python required |
| Transport | WebSocket real-time bidirectional bridge |

---

## Skills Demonstrated

- **Real-time systems** — Latency-sensitive audio pipeline, concurrent voice
  and chat paths running in parallel threads, WebSocket streaming with
  per-sentence TTS playback as tokens arrive
- **Multimodal AI** — Automatic routing between text and vision models based
  on input type detection; handles text, images, video, and URLs
- **API orchestration** — Six external services (Groq, ElevenLabs, OpenWakeWord,
  Breeze, Tavily, WebSocket) integrated into one coherent real-time system
- **Voice pipeline architecture** — Full wake word → STT → LLM → TTS chain
  with clean state machine (idle/listening/thinking/speaking)
- **WebGL / Shader programming** — Custom GLSL shaders for galaxy orb core,
  plasma shell, audio-reactive pulse waves, and full post-processing pipeline
- **Financial API integration** — Lightning Network payments via Breeze SDK,
  voice-triggered invoice creation and balance queries
- **Memory system design** — Persistent context with user-controlled long-term
  retention across both voice and text modalities
- **Desktop application packaging** — PyQt6 native window with embedded
  browser engine; ships as standalone .exe via PyInstaller

---

## Project Structure

```
chloe/
├── start_jarvis.py       # Main launcher — PyQt6 window + starts all services
├── jarvis.py             # Core brain — voice loop, chat, memory, Bitcoin
├── hud_server.py         # WebSocket bridge — connects HUD to backend
├── hud.html              # HUD interface — chat, avatar, state display
├── holo.html             # Standalone 3D orb viewer
├── holo-app.js           # Three.js scene setup and animation loop
├── holo-orb.js           # Galaxy orb shaders and mesh construction
├── holo-particles.js     # Particle system around the orb
├── holo-postfx.js        # Chromatic aberration + scanline post-processing
├── galaxy-orb.html       # Embeddable orb widget (iframe API)
├── requirements.txt      # Python dependencies
├── Jarvis.spec           # PyInstaller build config
├── _env                  # API keys (not committed to repo)
└── jarvis_icon.png       # App icon
```

---

## Setup & Installation

### Prerequisites
- Windows 10 or 11
- Python 3.11+
- ffmpeg — `winget install ffmpeg`

### Install

```bash
git clone https://github.com/contact-edwayne/Chloe.git
cd Chloe
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your own keys:

```bash
copy .env.example .env
```

The minimum required keys for chat to work:

```
GROQ_API_KEY=your_groq_key
```

Optional keys for the full experience:

```
# Premium voice (set USE_ELEVENLABS=1 to enable)
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=your_voice_id

# Bitcoin Lightning wallet
BREEZ_API_KEY=your_breez_key
```

See `.env.example` for the complete list of tunable settings.

### Run

```bash
cd chloe
venv\Scripts\activate
python start_jarvis.py
```

Or download the latest pre-built `.exe` from the [Releases](../../releases) page —
no Python installation required.

### Rebuild the .exe After Changes

```bash
pyinstaller Jarvis.spec
```

---

## Roadmap

- [ ] Expanded web search with multi-source synthesis and citations
- [ ] Persistent memory database (SQLite + vector embeddings)
- [ ] Calendar and task management integration
- [ ] Local model fallback for fully offline operation
- [ ] On-chain Bitcoin transaction support
- [ ] Mobile companion interface

---

## Built By

**Edward Wayne**

Built as a portfolio demonstration of real-time AI systems, multimodal
integration, financial API engineering, and holographic UI development.

Open to freelance projects and full-time roles in:
- Conversational AI / Voice AI Engineering
- AI Product Engineering
- Multimodal / Frontend AI Development
- Agentic Systems Engineering

📧 contact.edwayne@gmail.com · 💼 [linkedin.com/in/edward-wayne-4b74ab408](https://www.linkedin.com/in/edward-wayne-4b74ab408/) · 🐙 [github.com/contact-edwayne](https://github.com/contact-edwayne)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
