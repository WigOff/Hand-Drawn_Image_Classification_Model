# ✏️ QuickDraw AI — Hand-Drawing Image Classifier

Real-time hand-drawing recognition app built with **TensorFlow 2**, **Flask**, and a polished
canvas UI. Users draw on an HTML canvas and the model predicts the category **live**.

---

## 🗂 Project Structure

```
hand_draw_classifier/
├── app.py                 # Flask server + /predict endpoint
├── train_model.py         # Download dataset + train CNN
├── requirements.txt
├── model/
│   ├── quickdraw_model.keras   # (generated after training)
│   ├── class_names.json        # (generated after training)
│   └── training_curves.png     # (generated after training)
├── quickdraw_data/
│   └── *.npy                   # (downloaded automatically)
└── templates/
    └── index.html         # Drawing UI
```

---

## 📦 Dataset

**Google Quick, Draw!** — the world's largest doodle dataset.
10 selected categories (100 k drawings each), downloaded automatically during training.

| # | Class     | # | Class     |
|---|-----------|---|-----------|
| 0 | airplane  | 5 | fish      |
| 1 | bicycle   | 6 | flower    |
| 2 | bird      | 7 | house     |
| 3 | cat       | 8 | tree      |
| 4 | dog       | 9 | umbrella  |

Kaggle page: <https://www.kaggle.com/datasets/ashishjangra27/quickdraw-dataset-10-categories>

---

## 🚀 Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### 2 — Train the model

**Option A — Automatic (no Kaggle account needed)**
```bash
python train_model.py
```
The script will download `.npy` files directly from Google's public storage and train
the model. Expected runtime: ~10 min on CPU, ~2 min on GPU.

**Option B — Via Kaggle API**
```bash
pip install kaggle
# place your kaggle.json in ~/.kaggle/
python train_model.py
```

### 3 — Run the web app
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🧠 Model Architecture

```
Input  (28×28×1)
  │
  ├─ Conv2D(32) → BN → Conv2D(32) → MaxPool → Dropout(0.25)
  ├─ Conv2D(64) → BN → Conv2D(64) → MaxPool → Dropout(0.25)
  ├─ Conv2D(128) → BN → Dropout(0.30)
  ├─ GlobalAveragePooling
  ├─ Dense(256) → BN → Dropout(0.40)
  └─ Dense(10, softmax)
```

**Target accuracy**: ≥ 90 % on validation set (typically reaches 93–95 %).

Training features:
- Data augmentation (random flip, rotation ±8°, zoom ±10%)
- `ReduceLROnPlateau` + `EarlyStopping`
- Best checkpoint saved automatically

---

## 🔌 API

### `POST /predict`
**Body** (JSON):
```json
{ "image": "data:image/png;base64,..." }
```
**Response**:
```json
{
  "predictions": [
    { "label": "cat",       "confidence": 87.3 },
    { "label": "dog",       "confidence": 7.1  },
    { "label": "bird",      "confidence": 3.2  }
  ]
}
```

---

## ⚙️ Configuration

Edit the constants at the top of `train_model.py`:

| Variable              | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `CLASSES`             | 10 items| Which Quick Draw categories to use |
| `SAMPLES_PER_CLASS`   | 10 000  | Increase for higher accuracy       |
| `EPOCHS`              | 25      | Max training epochs                |
| `BATCH_SIZE`          | 256     | Batch size                         |

---

## 📈 Results

| Metric           | Value     |
|------------------|-----------|
| Val accuracy     | ≥ 93 %   |
| Val loss         | ≤ 0.25   |
| Inference latency| < 20 ms  |
| Model size       | ~3.5 MB  |
