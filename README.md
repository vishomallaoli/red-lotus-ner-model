
# eBay 2025 University Machine Learning Competition

## 📌 Overview
This repository contains our team’s submission for the **eBay 2025 University Machine Learning Competition**.  
The challenge focuses on **Named Entity Recognition (NER)** for **German e-commerce titles in the automotive domain**, with the goal of building a state-of-the-art system that exceeds eBay’s provided benchmark.

---

## 🎯 Objectives
- Develop robust **NER pipelines** for noisy, domain-specific German text.  
- Experiment with **state-of-the-art transformer models** for multilingual/domain-adapted learning.  
- Implement **custom preprocessing** (tokenization, lemmatization, normalization of car parts terminology).  
- Deliver reproducible **training, evaluation, and inference scripts**.  
- Submit results that aim to **win the competition** and secure a Summer 2026 internship with eBay.

---

## 👥 Team
- **Visho Malla Oli** – Tech Lead & Strategist  
- **Nitesh Subedi** – ML Research & Model Development  
- **Ishan Pathak** – Documentation & Model Execution  

---

## 🏗️ Project Structure
```

├── data/                # Raw and processed datasets (not tracked in repo)
├── notebooks/           # Jupyter notebooks for exploration & prototyping
├── src/                 # Source code for preprocessing, training, inference
│   ├── preprocessing/   # Tokenization, normalization, augmentation
│   ├── models/          # Model architectures, configs, and fine-tuning
│   ├── training/        # Training loops, schedulers, optimizers
│   ├── evaluation/      # Metrics, validation, error analysis
│   └── inference/       # Final prediction pipelines
├── configs/             # Config files for experiments
├── scripts/             # Utility scripts (data download, submission prep)
├── results/             # Model checkpoints, logs, and evaluation results
└── README.md            # Project documentation

```

---

## ⚙️ Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/vishomallaoli/red-lotus-ner-model.git
   cd red-lotus-ner-model
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Preprocess Data

```bash
python src/preprocessing/preprocess.py --input data/raw --output data/processed
```

### Train Model

```bash
python src/training/train.py --config configs/base_config.yaml
```

### Evaluate Model

```bash
python src/evaluation/evaluate.py --checkpoint results/checkpoint.pt
```

### Run Inference

```bash
python src/inference/predict.py --input data/test.csv --output results/predictions.csv
```

---

## 📊 Evaluation

* Metrics: **F1 score, Precision, Recall**
* Benchmark: eBay baseline (to be exceeded with our fine-tuned approach)

---

## 🧪 Experiments

We will track experiments using:

* **Weights & Biases (W\&B)** for logging
* **Hydra / YAML configs** for reproducibility

---

## 📅 Timeline

* **August–September 2025:** Data exploration, baseline models
* **October 2025:** Fine-tuning & error analysis
* **November 1, 2025:** Final submission

---

## 📜 License

MIT License – see [LICENSE](LICENSE) for details.
