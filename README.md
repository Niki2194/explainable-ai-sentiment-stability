# Explainable AI Stability in Noisy Social Media Sentiment Analysis

**M.Sc. Big Data & AI thesis project** | **Nikita Lakhani** | SRH University Heidelberg (Leipzig Campus)

## Overview

How reliable are explanations of a sentiment classifier when users introduce everyday noise into social media posts? This project compares **LIME** and **SHAP** explanations of a fine-tuned **BERT (`bert-base-uncased`)** model on the **TweetEval sentiment** dataset. It examines both **run-to-run reproducibility** and **explanation stability under text perturbations**.

## Highlights

- Fine-tuned a three-class BERT sentiment classifier (negative / neutral / positive).
- Evaluated **200 approximately class-balanced tweets** and **four perturbation types**: keyboard typos, emoji insertion, synonym replacement, and word deletion (**800 clean/noisy tweet pairs**).
- Compared explanation stability using **Jaccard overlap**, **sign consistency**, **L2 distance**, and **Spearman correlation**.
- Used **Wilcoxon signed-rank tests** for paired comparisons.
- Evaluated reproducibility over **three explanation runs per tweet** for each method.
- Created exploratory analyses and publication-oriented visualizations.

## Results

The following values are **reported by the saved notebook**, not independently reproduced for this repository.

| Evaluation | LIME | SHAP | Preferred method in this experiment |
|---|---:|---:|---|
| Run-to-run Jaccard consistency (higher is better) | 0.7750 | 1.0000 | SHAP |
| Noise stability: Jaccard overlap (higher is better) | 0.4098 | 0.3405 | LIME |
| Noise stability: sign consistency (higher is better) | 0.8715 | 0.7847 | LIME |
| Noise stability: L2 distance (lower is better) | 0.3132 | 0.3907 | LIME |
| Noise stability: Spearman correlation (higher is better) | 0.6951 | 0.6591 | LIME |

**BERT test-set Macro F1:** 69.09% (reported test size: 12,284). The notebook reports Wilcoxon p-values below 0.05 for all four noise-stability comparisons. Repeated-run determinism should not be assumed for other SHAP configurations or model environments.

**Interpretation:** In this specific setup, SHAP yielded more repeatable explanations, while LIME exhibited better average agreement between clean and perturbed texts across the evaluated metrics. These are distinct properties, and neither result implies universal superiority.

## Methodology

1. Load TweetEval sentiment splits and perform data-quality checks and exploratory analysis.
2. Fine-tune `bert-base-uncased` for three-class sentiment classification.
3. Select 200 test tweets across sentiment labels.
4. Generate four types of text perturbation per tweet.
5. Obtain LIME and SHAP token/word explanations on original and perturbed text.
6. Compare explanations across four stability metrics and run paired statistical tests.
7. Re-run explanations three times per original tweet and measure pairwise consistency.
8. Summarize results by noise type and sentiment class.

## Repository contents

```text
.
├── README.md
├── notebooks/
│   └── xai_stability_lime_shap.ipynb
├── requirements.txt
├── results/
│   └── README.md
├── figures/
│   └── README.md
└── .gitignore
```

## Run in Google Colab

1. Upload or open `notebooks/xai_stability_lime_shap.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select a GPU runtime for BERT fine-tuning and model inference.
3. Install the dependencies in `requirements.txt` or use the notebook's installation cells.
4. **Important:** The notebook was developed with Google Drive at `/content/drive/MyDrive/thesis_xai_stability`. Its session starter expects the trained model in `models/bert_finetuned` within that Drive folder. Adjust `BASE` to your Drive folder. If no trained model is present, run Section 3 to train it before using the explainers, and rerun the session starter and explainer setup afterward.
5. Follow the notebook's documented execution order. Training and full XAI analyses are computationally expensive, and checkpoints from the original Colab run are **not bundled** here.

**Reproducibility note:** The repository includes the primary notebook but not the private Drive model weights, CSV checkpoints, or exported figures. Some package APIs and Colab defaults can change; pinned versions and a fresh end-to-end execution are recommended before claiming full reproducibility. The notebook was cleaned of execution outputs and exploratory scratch cells for portfolio presentation; original reported results are retained in its conclusions.

## Data

Dataset: [TweetEval (`cardiffnlp/tweet_eval`, sentiment configuration)](https://huggingface.co/datasets/cardiffnlp/tweet_eval). Data is loaded through Hugging Face rather than redistributed here.

## Technologies

Python · PyTorch · Hugging Face Transformers / Datasets · BERT · LIME · SHAP · scikit-learn · pandas · NumPy · SciPy · NLTK · nlpaug · Plotly · Matplotlib · Seaborn

## Limitations

- Stability study covers 200 sampled test tweets, one classifier architecture, and one sentiment dataset.
- Perturbations are synthetic rather than a comprehensive sample of naturally occurring social media errors.
- Metrics depend on explanation tokenization, feature mapping, and implementation details.
- A model's predictive performance and the stability of its explanations answer different questions.
- Results were extracted from the notebook's saved summaries; no independent rerun was performed for this GitHub package.

## Author

**Nikita Lakhani**  
M.Sc. Big Data & AI, SRH University Heidelberg (Leipzig Campus)

*Academic research portfolio project.*
