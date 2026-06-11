# Speech-to-Speech Translation: Cascaded vs. Textless Benchmarking

This repository contains an evaluation framework comparing a standard Tri-Stage Cascaded Network against a Unified Textless Topology for Speech-to-Speech Translation (S2ST). 

The primary objective of this project is to evaluate performance latency and implement **Latent PII Surgery**—a feature-level masking mechanism designed to redact sensitive numeric data (e.g., account numbers) within the hidden feature space. This approach explores methods to support data privacy standards (such as GDPR, CCPA, and HIPAA) without generating plaintext strings in system memory.

🌐 **Live Interactive Workspace:** [Launch on Hugging Face Spaces](https://huggingface.co/spaces/mugil6/audio_ai)

---

## 🏗 Architecture Overview

Traditional translation systems can experience alignment instability when converting dense numeric payloads across different dialects. The textless implementation evaluated in this repository addresses this by bypassing text conversion entirely.

### 1. Traditional Cascaded Pipeline (Baseline)
```text
[Audio] ➔ Whisper ASR ➔ Plaintext (Memory) ➔ MarianMT ➔ VITS TTS ➔ [Output]

Observation: Processes intermediate text, which introduces latency and creates plaintext records of potentially sensitive transcription data in system memory.

2. Textless Topology with Latent Surgery (Proposed)

[Audio] ➔ Log-Mel Spectrogram ➔ Latent PII Masking (H = 0.0) ➔ Discrete Units ➔ Vocoder ➔ [Output]

Observation: Bypasses intermediate text generation. Data masking is applied directly to the tensor matrix before discrete unit quantization.

Empirical Execution Results

Using hardware-level clock timing on a standard GPU instance, we tracked the compute latency of both pipelines. The results below reflect operations on uniform audio tracks containing standard speech and numeric identifiers.

Evaluation Axis,Cascaded Baseline,Textless Topology (Ours),Observation
Intermediate Format,Plaintext Strings,Discrete Units,Bypasses text registers
Latency (Scenario 1: 4.78s),4.366s,2.595s,Reduced compute time
Latency (Scenario 2: 3.94s),3.427s,1.199s,Reduced compute time
Privacy Compliance,Plaintext Logged,Compliant (Masked),Applies latent redaction

(Note: Live deployment latencies on free-tier CPU spaces will reflect higher framework and network overhead compared to raw GPU compute timings).

Scaling for Production Environments

To adapt this benchmarking prototype for live, concurrent usage, the following infrastructure adjustments are recommended:

    gRPC/WebSocket Streaming: Migrate from static file uploads to overlapping audio packet streams (e.g., 200ms windows) to reduce ingestion bottlenecks.

    Dynamic Batching: Deploy the textless model using a serving layer like NVIDIA Triton Inference Server with dynamic batching enabled to optimize GPU core utilization for concurrent requests.

    Ephemeral Storage (tmpfs): Run inference container pods on RAM-backed file systems to ensure latent tensors and spectrogram matrices are cleared from memory post-execution.
Local Development

Clone the repository and install the required dependencies to run the Gradio workspace locally:

git clone [https://github.com/Mugil6/translation_ai](https://github.com/Mugil6/translation_ai)
cd translation_ai
pip install -r requirements.txt
python app.py

📜 License

Copyright (c) 2026 Mugilan. All Rights Reserved.

