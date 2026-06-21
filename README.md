# 🌾 CropGuard AI

**AI-powered crop protection — detect plant diseases and crop pests instantly from a single photo.**

CropGuard AI is a dual-model computer vision app that helps farmers, agronomists, and researchers diagnose plant health issues in seconds. Upload a leaf photo to detect diseases across 14 plant species, or upload a field image to identify one of 102 pest and insect types — all powered by deep learning, served through a clean Streamlit interface.

---

## ✨ Features

- 🌿 **Plant Disease Detection** — Identifies 38 disease/health classes across 14 plant species using a custom CNN
- 🐛 **Crop Pest Detection** — Identifies 102 pest and insect types across rice, wheat, maize, vegetables, and fruits using EfficientNetB0
- ⚡ **Instant inference** — Results in under 2 seconds per image
- 📊 **Confidence-ranked predictions** — See top predictions with confidence scores, not just a single guess
- 📚 **Reference library** — Browse every detectable disease and pest class
- 🎨 **Polished, responsive UI** — Built entirely in Streamlit with custom theming

---

## 🧠 Models

| | Plant Disease Model | Crop Pest Model |
|---|---|---|
| **Architecture** | 5-block CNN (Conv2D + BatchNorm + MaxPooling → GAP → Dense) | EfficientNetB0 (fine-tuned) |
| **Classes** | 38 | 102 |
| **Input size** | 256 × 256 | 224 × 224 |
| **Dataset** | PlantVillage (87,000+ images) | 102-class crop pest dataset |
| **Training** | Adam, CategoricalCrossentropy, EarlyStopping, ReduceLROnPlateau | Transfer learning, EfficientNet preprocessing |

---

## 🛠️ Tech Stack

- **Frontend / App:** Streamlit
- **Deep Learning:** TensorFlow, Keras 3
- **Image Processing:** Pillow, NumPy
- **Language:** Python

---



## 📁 Project Structure

```
cropguard-ai/
├── main.py               # Streamlit app (UI + inference logic)
├── model.keras            # Plant disease model
├── best_model.keras       # Crop pest model
├── class_mapping.json     # Pest label mapping
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

- [ ] **AI-powered remedy API** — integrate a solution/treatment-recommendation API so that once a disease or pest is detected, the app also returns suggested treatment, prevention steps, and care guidance — not just the diagnosis
- [ ] Multi-image / batch upload support
- [ ] Downloadable PDF diagnosis report
- [ ] Mobile camera capture support
- [ ] Model confidence calibration & explainability (Grad-CAM visualizations)
- [ ] Multilingual support for regional farmers

---


