# Personality Detection — Introvert vs Extrovert
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](
(https://personality-detection-bcsmog26e6uqc8t57xp3vc.streamlit.app))
## Overview
A real-time NLP classifier that predicts personality type
(Introvert / Extrovert) from text input. Built with Python,
scikit-learn, and deployed on Streamlit Cloud.
## Results
| Metric | Score |
|-----------|--------|
| Accuracy | 77.1% |
| F1 Score | 0.78 |
![Confusion Matrix](<img width="900" height="750" alt="image" src="https://github.com/user-attachments/assets/1676762f-e5df-4400-a0f8-10997be76ee2" />
)
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
git clone https://github.com/YOUR_USERNAME/personality-detection
cd personality-detection
pip install -r requirements.txt
python src/train.py # trains and saves model
streamlit run app.py # launches web app
```
## Project Structure
```
personality-detection/
 app.py # Streamlit app
 src/
|  preprocess.py # text cleaning
|  train.py # model training + metrics
 data/
|  model.pkl
|  vectorizer.pkl
|  confusion_matrix.png
 requirements.txt
```
