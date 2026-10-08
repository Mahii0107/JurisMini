# ⚖️ JurisMini
### Neobrutalist Legal SLM Intelligence: Benchmarking TinyLlama, Qwen, and Phi on Indian Court Judgments

[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg)](https://vite.dev)
[![Style](https://img.shields.io/badge/Aesthetic-Neobrutalist%20%26%20Comic-FACC15.svg)](#)
[![Backend](https://img.shields.io/badge/Backend-Python%20Flask%20API-black.svg)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-InLegalNER-green.svg)](https://huggingface.co/datasets/opennyaiorg/InLegalNER)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Device%20SLM-success.svg)](#)

---

## 📌 Executive Summary

**JurisMini** is a Neobrutalist, high-contrast AI research and deployment platform designed to extract critical judicial entities from Indian court judgments using **Small Language Models (SLMs)** under 4 billion parameters. 

Rather than relying on expensive and privacy-sensitive commercial cloud APIs, this project fine-tunes and benchmarks three open-weight SLMs on the official **InLegalNER** dataset using **QLoRA (4-bit quantization)**:
1. **TinyLlama-1.1B** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
2. **Qwen2.5-1.5B** (`Qwen/Qwen2.5-1.5B`)
3. **Phi-3-mini** (`microsoft/Phi-3-mini-4k-instruct`)

---

## 🎨 Neobrutalist & Vintage Comic Aesthetic

The web interface is styled with a strict Neobrutalist and vintage comic design system:

* **Retro Pop Color Palette**: Pastel yellow background (`#FFF3CD`) with subtle dot-grid texture, stark black borders, and vibrant accents of electric blue (`#2563EB`), retro orange (`#FF6B00`), and canary yellow (`#FACC15`).
* **Solid Borders & Hard Drop Shadows**: 3px/3.5px solid black borders with offset drop shadows (`4px 4px 0px #000` to `8px 8px 0px #000`) instead of blurry glows.
* **Tactile Buttons & Interactive Cards**: Paper cutout dossiers, angled sticker stamps, and active press-down states (`transform: translate(2px, 2px)`).
* **High-Impact Typography**: Heavy display fonts (**Space Grotesk** and **Syne**) paired with monospace metadata (**Space Mono**) and readable body copy (**Public Sans**).
* **Case Dossier & Speech Bubbles**: Structured legal breakdown highlighting the 1-sentence ratio, presiding bench, litigating parties, and cited statutes.

---

## 🚀 How to Run the App

### Option 1: Double-Click (Windows)
Double-click [`run_app.bat`](file:///c:/Users/Mrunmayee%20Potdar/Downloads/AISC_mpr/run_app.bat) in the project folder.

### Option 2: Command Line (PowerShell / Terminal)
```powershell
# 1. Start Python Legal AI API Backend
python server.py

# 2. In another terminal, start React + Vite frontend
cd frontend
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your web browser.

---

## 📊 Benchmark Results on Official Test Set (4,501 Documents)

All models were evaluated on the official test set (13,365 human-annotated entities across 14 legal entity types):

| Configuration | Exact Precision | Exact Recall | **Exact F1** | Macro F1 | Overlap F1 | Train Time (8GB GPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-1.5B (Causal) | 0.574 | 0.605 | **0.589** | 0.588 | 0.784 | ~26 min |
| Qwen2.5-1.5B (Causal + Snap) | 0.632 | 0.637 | **0.635** | 0.632 | 0.797 | — |
| Phi-3-mini-3.8B (Causal) | 0.556 | 0.720 | **0.627** | 0.614 | 0.820 | ~62 min |
| Phi-3-mini-3.8B (Causal + Snap) | 0.632 | 0.771 | **0.695** | 0.680 | 0.841 | — |
| TinyLlama-1.1B (Causal + Snap) | 0.646 | 0.775 | **0.704** | 0.687 | 0.847 | ~25 min |
| TinyLlama-1.1B (Bidirectional) | 0.738 | 0.819 | **0.777** | 0.767 | 0.895 | ~24 min |
| 🏆 **TinyLlama-1.1B (Bidirectional + Snap)** | **0.773** | **0.833** | **0.802** | **0.792** | **0.903** | **~24 min** |

### Key Research Insights:
1. **Bidirectional Attention is Critical for Legal NER (+13% F1)**: Decoder-only causal attention cannot look ahead to see titles like *"Advocate for the petitioner"*, causing role misclassifications. Switching attention to bidirectional unlocks dramatic accuracy gains.
2. **Word-Boundary Snapping (+5% F1)**: Heuristic post-processing repairs partial token boundaries without needing retraining.
3. **Parameter Efficiency**: TinyLlama-1.1B out-performed the 3.8B parameter Phi-3 model while requiring less than half the training time and memory footprint.

---

## 📁 Repository Structure

```text
AISC_mpr/
├── run_app.bat                                  # Windows one-click launcher
├── README.md                                    # Project documentation
└── New folder (2)/
    ├── app.py                                   # Redesigned Jan Nyaya Streamlit web app
    ├── Indian_Legal_NER_SLM_Comparison.ipynb    # Main research & training notebook (Unchanged)
    ├── NOTEBOOK_GUIDE.md                        # Complete technical guide for the notebook
    ├── runs/                                    # Initial training logs and results
    ├── runs_v2/                                 # Bidirectional attention ablation logs
    ├── final_legal_ner_model/                   # Trained adapter weights (v1)
    └── final_legal_ner_model_v2/                # Final best model adapter weights (TinyLlama bidir)
```

---

## ⚖️ 14 Legal Entity Types Recognized
* `COURT`, `JUDGE`, `PETITIONER`, `RESPONDENT`, `LAWYER`, `CASE_NUMBER`, `STATUTE`, `PROVISION`, `PRECEDENT`, `DATE`, `WITNESS`, `OTHER_PERSON`, `ORG`, `GPE`.
