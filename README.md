# SHL Audio Grammar Scoring

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-0.806-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A deep learning pipeline to automatically score the grammatical quality of spoken audio samples. Built for the **SHL Labs - Audio Grammar Scoring Kaggle Challenge**.

---

## Results

| Version | Public Score (Pearson r) | Approach |
|---------|--------------------------|----------|
| V1 | 0.752 | wav2vec2-base, basic pooling |
| V6 | 0.806 | wav2vec2-large, multi-layer pooling, stacked ensemble |
| V7 | 0.457 | + handcrafted features, 10-fold CV, SVR |

---

## Approach

### 1. Feature Extraction
- **wav2vec2-large-960h** (Facebook) pretrained transformer for deep speech representations
- **Weighted average of ALL hidden layers** (higher weights for later layers) with mean + std pooling over time
- **Handcrafted acoustic features:**
  - 40 MFCCs + Delta MFCCs (mean & std)
  - Spectral centroid, bandwidth, rolloff
  - Chroma features (12 bins)
  - Mel spectrogram statistics (64 mel bands)
  - Pitch (F0) via PYIN — mean, std, voiced ratio
  - Zero crossing rate, RMS energy
  - Speech rate (onset count per second)

### 2. Preprocessing
- Silence trimming (`librosa.effects.trim`, top_db=20)
- Amplitude normalization (unit std)
- Audio capped at 30 seconds
- `StandardScaler` normalization
- PCA dimensionality reduction (300 components)

### 3. Stacked Ensemble
- **Base models** (10-fold cross-validation with OOF predictions):
  - LightGBM (2000 trees, lr=0.02, early stopping)
  - XGBoost (2000 trees, lr=0.02)
  - Ridge Regression (alpha=10)
  - SVR (RBF kernel, C=10)
- **Meta-learner:** Ridge Regression on OOF predictions
- Final predictions clipped to training label range (continuous, not rounded)

---

## Project Structure

```
SHL-Audio-Grammar-Scoring/
├── notebook.ipynb          # Main pipeline notebook
├── README.md
├── .gitignore
└── LICENSE
```

---

## Requirements

```
torch
transformers
librosa
numpy
pandas
scikit-learn
lightgbm
xgboost
tqdm
scipy
```

Install with:
```bash
pip install torch transformers librosa numpy pandas scikit-learn lightgbm xgboost tqdm scipy
```

---

## How to Run

1. Download the dataset from [Kaggle Competition](https://www.kaggle.com/competitions/shl-audio-scoring-challenge)
2. Open `notebook.ipynb` in Kaggle or Google Colab
3. Run all cells sequentially
4. Submit `submission.csv` to the competition

**Kaggle Notebook:** [notebook60f03c62ae](https://www.kaggle.com/code/yashkohinkar/notebook60f03c62ae)

**Google Colab:** [Open in Colab](https://colab.research.google.com/drive/1-sKF7oPL2Oe1u4hnjhUu7Q4TKZzL1sE2?usp=sharing)

---

## Key Design Decisions

- **Why wav2vec2-large over base?** The large model (24 transformer layers, 1024-dim) captures richer phonemic and prosodic patterns essential for grammar scoring.
- **Why weighted layer averaging?** Later layers encode higher-level linguistic features; early layers capture low-level acoustics. Weighting linearly (0.5→1.0) leverages both.
- **Why handcrafted features?** Pitch, speech rate, and MFCCs directly relate to fluency and grammatical confidence — complementing the learned representations.
- **Why not round predictions?** The metric is Pearson correlation, which performs better with continuous predictions than integer-rounded ones.
- **Why 10-fold over 5-fold?** More folds = more stable OOF predictions = better meta-learner training, especially with smaller datasets.

---

## Competition

**[SHL Labs - Audio Grammar Scoring Challenge](https://www.kaggle.com/competitions/shl-audio-scoring-challenge)**

The task is to predict a grammar score (continuous) from spoken audio samples. Evaluation metric: **Pearson correlation coefficient**.

---

## Author

**Yash Kohinkar**
- Kaggle: [@yashkohinkar](https://www.kaggle.com/yashkohinkar)
- GitHub: [@yashkohinkar](https://github.com/yashkohinkar)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
