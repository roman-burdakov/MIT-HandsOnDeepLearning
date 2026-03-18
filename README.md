# HODL (Hands-on Deep Learning) — 15.773

> MIT Sloan School of Management | Spring 2026
> Instructors: Rama Ramakrishnan

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://tldrlegal.com/license/mit-license#fulltext)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Framework-Keras-red.svg)](https://keras.io/)
[![Canvas](https://img.shields.io/badge/Course-Canvas-orange.svg)](https://canvas.mit.edu/courses/37731)

---

## 🗂️ Repository Structure

```
MIT-HandsOnDeepLearning
/
│
├── 📁 examples/               # In-class code examples by topic
├── 📁 homeworks/              # Homework assignments & starter notebooks
├── 📁 recitations/            # Recitation support materials
├── 📁 final_project/          # Final project
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🗓️ Course Schedule & Examples

| Topic | Key Concepts |
|-------|--------------|
| Introduction to Neural Networks & Deep Learning | Layers, activations, forward pass |
| Training Deep NNs | Loss functions, SGD, regularization |
| Tabular Data & Hyperparameter Optimization | Keras Tuner, tabular pipelines |
| NLP: Tokenization & Bag of Words | Text vectorization, BoW models |
| Transformers (1/3) | Attention mechanism |
| Transformers (2/3) | Multi-head attention, positional encoding |
| Transformers (3/3) | Encoder-decoder, fine-tuning |
| Computer Vision: Vision Transformer | ViT architecture, image patches |
| LLMs: Pretraining | GPT architecture, next-token prediction |
| LLMs: Post-Training & RAG | SFT, RLHF, retrieval augmented generation |
| LLMs: Fine-Tuning & Agents | LoRA, PEFT, agent frameworks |
| Text-to-Image Diffusion Models | Stable Diffusion, DALL-E concepts |

---

## 📝 Homework Assignments

Three **individual** homework assignments to get us started.

**HW1 — Neural Networks & NLP Basics**
Neural network fundamentals, tabular data prediction, hyperparameter optimization, tokenization

**HW2 — Transformers & Computer Vision**
Transformer architecture, NLP sequence modeling, Vision Transformer (ViT)

**HW3 — Large Language Models**
LLM pretraining, RLHF, RAG, parameter-efficient fine-tuning

---

## Final Project
### 🎬 Box Office Prediction

> **"Box Office Prediction v2: Multimodal Forecasting of Opening Weekend Revenue"**

**Problem:** Forecast U.S. domestic opening weekend box office *before release* using pre-release audience signals from YouTube and structured movie metadata — framed as a supervised regression problem evaluated with R² and MAE in log space. The core setup deliberately excludes number of opening theaters to simulate a true pre-release prediction scenario.

**Dataset** (344 movies) assembled via Box Office Mojo scraping, TMDB API, and YouTube Data API v3. Each movie is matched to its primary YouTube trailer; comments are collected and windowed into 4 time bins relative to trailer publish date.

**Feature pipeline:**

| Source | Features |
|--------|----------|
| Structured metadata | Release timing, genre, distributor, rating, runtime, franchise proxy, pandemic regime |
| Pre-trained embeddings | `all-mpnet-base-v2` sentence embeddings on title + tagline + overview |
| YouTube trailer text | Trailer title, description, tags encoded via same embedder |
| YouTube comments | Early-window comment embeddings across 4 time bins; engagement stats; RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) sentiment with trend slope |
| Retrieval features | Similarity-based features to historically comparable titles |

**Modeling approach:**
Four model families were trained and compared using a 75/25 random train/test split with stratification on log-revenue. All neural models use 5-seed ensembles with early stopping and learning rate reduction:

| Model | Test R²(log) | Test MAE(log) |
|-------|-------------|--------------|
| 🥇 ResNet-style Tab+Text NN (3-seed ensemble) | **0.586** | **0.489** |
| 🥈 XGBoost Tab+Text (tuned, cross-validated) | 0.577 | 0.495 |
| Late-Fusion Transformer (text branch + tab MLP) | 0.535 | 0.532 |
| Tiny Text-Only MLP (baseline) | 0.470 | 0.540 |

**Key findings:**
- The **ResNet-style multimodal NN** — two residual blocks, width 120, dropout 0.15, L2 1.5e-3, batch size 8 — was the final winner across both log-space and dollar-space metrics
- **Feature engineering improvements** (better trailer matching, comment bin features, franchise proxies, pandemic regime flags) contributed more to performance gains than architecture changes
- The strongest individual tabular predictors were `runtime_min` and `is_four_quadrant_family_like`; social/sentiment features (`yt_comment_unique_ratio_early`, `yt_sent_mean_early`) added incremental signal
- **Late-fusion Transformer** — separate text Transformer branch fused with a tabular MLP — outperformed all scalar-tab tokenized Transformer designs
- **XGBoost** nearly matched ResNet in log-space but underperformed in dollar-space calibration on large-opening blockbusters
- Remaining errors concentrate on franchise/event releases and movies with sparse or atypical pre-release social signals

**Notebook** runs in both Google Colab and local environments with auto-configured data paths.

---

## 📚 Textbook

**[Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-third-edition)** — François Chollet, 3rd Ed. (2025) · **Free at the link**

---

## 📜 License

All materials and project submissions are under the **[MIT License](./LICENSE)**.
