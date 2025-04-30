# Sorting Hat 2.0 — Hogwarts-House Classification with a Classical NLP Pipeline

> **Predict any Harry-Potter character’s house (Gryffindor, Slytherin,  
> Ravenclaw, Hufflepuff) using lightweight, fully interpretable models.**

---

## 📂 Project Structure

```
.
├── data/
│   ├── raw_books/              # plain-text novels (not tracked)
│   ├── house_mapping.json      # {name : canonical_house}
│   └── mentions.csv            # harvested & labelled names
├── notebooks/
│   └── 01_EDA_and_Baselines.ipynb
├── src/
│   ├── harvest.py              # SpaCy NER → mentions.csv
│   ├── features.py             # TF-IDF + χ² selector
│   ├── train.py                # nested-CV, ensemble, metrics
│   └── inference.py            # CLI prediction script
├── environment.yml             # Conda (Python 3.11, spaCy, scikit-learn)
├── HP_House_Report.tex         # 8-page article-style report
└── README.md                   # you are here
```

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-user>/HP_House_Classifier.git
cd HP_House_Classifier
conda env create -f environment.yml
conda activate hogwarts
python -m spacy download en_core_web_lg   # if not auto-installed
```

---

## ▶️ Quick Start

```bash
# 1️⃣ harvest entities   (≈ 10 s)
python src/harvest.py  --books data/raw_books

# 2️⃣ train & evaluate   (≈ 30 s CPU)
python src/train.py    --mentions data/mentions.csv \
                       --out model.pkl

# 3️⃣ try it interactively
python src/inference.py --model model.pkl --name "Luna Lovegood"
```

---

## 📈 Key Results

| Metric (test set, 1 231 names) | Value |
|--------------------------------|-------|
| Accuracy                       | **0.901** |
| Macro F<sub>1</sub>            | **0.886** |
| Model size                     | 670 kB |
| Median latency (RPi 4)         | 2.8 ms |

*Ensemble = Multinomial NB + Linear SVM + Logistic Regression,  
weighted soft-vote (0.15 : 0.45 : 0.40).*

---

## 🔍 Insights

* **Lexical cues dominate**: “weasley”, “potter” ⇒ Gryffindor; “malfoy”, “crabbe” ⇒ Slytherin.  
* **Ravenclaw ↔ Hufflepuff overlap** causes 75 % of residual errors.  
* Character 3-5-gram TF-IDF boosts recall for rare names without inflating the feature space.  
* Full pipeline runs on edge devices (< 1 MB, < 5 ms).

---

## 🛠  Commands

| Task | Command |
|------|---------|
| Rebuild everything (harvest ➜ train ➜ report) | `make all` |
| Single prediction | `python src/inference.py -m model.pkl -n "Neville Longbottom"` |
| 5×5 nested CV grid-search | `python src/train.py --cv grid` |
| Lint & unit tests | `make check` |

---

## 🔮 Roadmap

- [ ] Add role-based tokens (“professor”, “captain”) to cut staff errors  
- [ ] Distil ensemble into a single fastText model for micro-controllers  
- [ ] Character-level CNN baseline for comparison (still < 1 MB)  

---

## 📄 License

MIT — do what you wish, but cite the original *Harry-Potter* work and this repo if you build on it.  

*© 2024 Mahla Entezari — Shahid Beheshti University*