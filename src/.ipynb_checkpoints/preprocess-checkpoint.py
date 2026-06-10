import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove MBTI type mentions (to avoid data leakage)
    text = re.sub(
        r'\b(infp|infj|intp|intj|enfp|enfj|entp|entj|estp|estj|esfp|esfj|isfp|isfj|istp|istj)\b',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Lowercase
    text = text.lower().strip()

    return text


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    df['label'] = df['type'].apply(
        lambda x: 1 if x[0] == 'I' else 0  # 1 = Introvert, 0 = Extrovert
    )

    df['clean_posts'] = df['posts'].apply(clean_text)

    # Remove empty rows after cleaning
    df = df[df['clean_posts'].str.strip() != '']

    return df


def split_data(df):
    X = df['clean_posts']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test