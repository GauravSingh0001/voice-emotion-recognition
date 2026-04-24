# Aura · Emotion Intelligence Frontend

This repository contains the client-side code for the **Aura Emotion AI** platform, a real-time emotion intelligence visualizer powered by Groq and Supabase. The frontend is built with vanilla JavaScript, HTML, and CSS to prioritize high performance, low latency, and a premium visual aesthetic.

## 🗂 File Structure and Brief Descriptions

### Core Application
- **`index.html`**  
  The main entry point for the application. Contains the UI structure for the real-time visualizer, onboarding flows, microphone testing overlays, and session control elements.

- **`main.js`**  
  The core frontend logic controller. It manages session state, connects to the backend and Supabase Realtime for verdict streaming, drives UI elements (progress rings, toasts, glassmorphic cards), tracks silence detection to trigger rendering, and handles manual utterance finalizations.

- **`result.html`**  
  The detailed analysis report dashboard. Displays the generated metrics—such as transcripts, dominant emotions, emotion breakdown, and intensity timelines. It includes a robust offline fallback mode that loads previous results from `sessionStorage` when the backend is unavailable.

- **`style.css`**  
  The global stylesheet implementing Aura's unique aesthetic. Blends sleek glassmorphism with neobrutalist accents, animations, and responsive layouts for a premium user experience.

### Visual & Integration Modules
- **`particle-aura.js`**  
  A Three.js-powered visual component. Renders the interactive, 3D fluid particle orb that reacts dynamically in real-time to the user's voice volume and recognized emotional state. Includes performance optimizations for mobile environments.

- **`emotion-config.js`**  
  The centralized configuration file that maps specific emotions (Happy, Sad, Surprised, etc.) to coherent visual design tokens like colors, frequencies, base radii, and glowing effects.

- **`livekit.js`**  
  Dedicated module handling the WebRTC connection using LiveKit. It acquires the user's microphone stream and manages real-time audio transmission directly to the backend analysis pipeline.

### Deployment & Infrastructure
- **`scripts/inject-env.js`**  
  A Node.js build script. Run prior to deployment to scan `.env` files and securely replace placeholders (`__BACKEND_URL__`, etc.) within `index.html` and `result.html` with real environment values.

- **`service-worker.js`**  
  A Progressive Web App (PWA) service worker. Implements caching strategies (Network-first locally, Cache-first in production) to cache static assets, speeding up load times and supporting offline demo functionality.

- **`manifest.json`**  
  Standard Web App Manifest defining the PWA's installable properties, icons, and theme colors.

- **`tests/`**  
  Directory designated for automated and manual test scripts to validate the frontend's interactions.

## 🚀 Quick Start
To run the frontend locally:
1. Ensure your `.env` is configured properly.
2. Run `node scripts/inject-env.js` to inject environment variables into the HTML views.
3. Serve the directory with a static HTTP server (e.g., `npx serve` or `python -m http.server 3000`).
