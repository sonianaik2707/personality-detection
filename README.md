# Personality Detection — Introvert vs Extrovert
### Live Demo
https://personality-detection-bcsmog26e6uqc8t57xp3vc.streamlit.app

[Open Streamlit App](https://personality-detection-bcsmog26e6uqc8t57xp3vc.streamlit.app)
## Overview
A real-time NLP classifier that predicts personality type
(Introvert / Extrovert) from text input. Built with Python,
scikit-learn, and deployed on Streamlit Cloud.
## Results
![Confusion Matrix](images/confusion_matrix.png)

| Metric | Value |
|---|---|
| Accuracy | 77.1% |
| Macro F1 | 0.70 |
| Weighted F1 | 0.78 |
| Extrovert Recall | 0.65 |
| Extrovert Precision | 0.50 |
| Introvert Recall | 0.81 |
| Introvert Precision | 0.88 |

**Baseline comparison:** The dataset is imbalanced (77% Introvert, 23% Extrovert). 
A naive baseline that always predicts "Introvert" would score 77.1% accuracy but 
0% recall on the Extrovert class — meaning it would never correctly identify a 
single extrovert. This model achieves 65% recall on Extrovert, showing it has 
learned real signal from the text rather than exploiting class imbalance.

Because of this imbalance, **macro F1 (0.70)** is reported alongside accuracy as 
a more honest measure of performance, since it weighs both classes equally 
rather than being dominated by the majority class.

## Limitations

- **Dataset imbalance**: 77% of samples are Introvert, which inflates raw 
  accuracy as a metric. Per-class recall/precision are reported to give a 
  truer picture.
- **Precision on Extrovert (0.50)** is currently the weakest point — half of 
  the model's Extrovert predictions are false positives. This is a known 
  tradeoff of using `class_weight='balanced'`, which improves recall at some 
  cost to precision.
- **MBTI as a framework** lacks strong psychometric validation in psychology 
  research and is not a clinically recognized measure of personality. This 
  project uses it as a text classification task, not as a claim about 
  personality science.
- **Self-reported labels**: the dataset's MBTI labels come from users' 
  self-identification, not validated assessments, so label noise is possible.
## Tech Stack
- Python, pandas, scikit-learn
- TF-IDF vectorisation (5000 features, bigrams)
- Logistic Regression with class balancing
- Streamlit for web deployment
## Dataset
MBTI Personality Type Dataset —
[Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type)
8,675 social media posts labelled with 16 MBTI personality types.
Simplified to binary: Introvert (I*) vs Extrovert (E*).
## Run Locally
```bash
git clone https://github.com/sonianaik2707/personality-detection.git
cd personality-detection
pip install -r requirements.txt
python src/train.py   # trains the model and saves model.pkl + vectorizer.pkl
streamlit run src/app.py   # launches the Streamlit web app
```
## Project Structure
```
personality-detection/
├── data/
│   └── mbti_1.csv
│
├── images/
│   ├── class_dist.png
│   └── confusion_matrix.png
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── app.py
│   ├── preprocess.py
│   ├── train.py
│   └── tf_idf_vectorization.py
│
├── requirements.txt
└── README.md
```
```
