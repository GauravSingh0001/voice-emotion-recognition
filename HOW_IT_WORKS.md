# How Aura Works: A Simple Guide

Welcome to Aura! This document explains exactly how our Emotion AI platform works, from the moment you start speaking to the moment an emotion appears on your screen. 

We've broken this down into a simple step-by-step journey so you can understand exactly where your data goes and how it is used.

---

## The Journey of Your Voice

Imagine you are having a conversation with an incredibly perceptive listener. Here is how Aura processes what you say.

### Step 1: Listening (The Ear)
* **What happens:** You click "Start" on the website and begin speaking into your microphone.
* **Incoming Data:** Your voice (audio) enters your browser.
* **Outgoing Data:** Your browser sends this continuous audio stream over a highly optimized, real-time "highway" (called LiveKit) directly to our backend server. 

### Step 2: Quick Reactions (The Reflex)
* **What happens:** As your audio arrives at our server, the system acts like a reflex. It chops your audio into tiny, half-second slices. 
* **The Process:** A small, incredibly fast "local brain" looks at *how* you are speaking—the pitch, the volume, the tone—without understanding the actual words. It makes a split-second guess (e.g., "This sounds angry!").
* **Why it matters:** This helps the system understand the acoustic emotion of your voice as it's happening, much like reading someone's body language.

### Step 3: Understanding the Words (The Secretary)
* **What happens:** The system is constantly listening for you to take a breath or pause. Once you stop speaking for a moment, it assumes you've finished a thought (an "utterance").
* **Outgoing Data:** The server packages that complete audio clip and sends it to a lightning-fast transcription service (Groq Whisper).
* **Incoming Data:** The service quickly returns the exact text of what you just said (e.g., *"Oh, great. Another flat tire."*).

### Step 4: The Final Judge (The Thinker)
* **What happens:** Now the system has two pieces of the puzzle: the written words and the "reflex" guesses about your tone of voice.
* **Outgoing Data:** The server sends both the text and the tone history to a massive, highly intelligent "supercomputer brain" (an AI called Llama). 
* **The Process:** The Judge looks at everything contextually. If the words are *"Oh, great"*, but the tone was frustrated and angry, the Judge realizes you are being **Sarcastic**.
* **Incoming Data:** The Judge sends back the final verdict: The emotion, a confidence score, and a brief explanation of *why* it chose that emotion.

### Step 5: The Result (The Display)
* **What happens:** The backend needs to get this final verdict back to your screen.
* **Outgoing Data:** The server saves the final verdict securely into our cloud database (Supabase).
* **Incoming Data (to your screen):** The database is "real-time," meaning the exact second the verdict is saved, it pushes a live notification directly to your web browser. Your screen updates instantly, showing you a beautiful "Emotion Card" with your results.

---

## Summary of Data Flow

If you want to look at it strictly in terms of data moving in and out of the main system:

**1. Data Coming IN from the User:**
- Continuous live microphone audio.
- (Optional) Uploaded `.wav` audio files.

**2. Data Moving OUT to Cloud Services:**
- Audio clips sent to the AI Transcriber (Groq Whisper).
- Text and tone data sent to the AI Judge (Groq Llama).
- Final results saved to the Database (Supabase).

**3. Data Going BACK to the User:**
- The final, processed emotional verdict (Emotion type, confidence score, transcript, and reasoning) pushed directly to their screen.

---
*No audio is permanently stored without permission, and the system is designed to be as fast and private as possible.*
