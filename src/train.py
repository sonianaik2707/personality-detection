import pickle
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

sys.path.append(os.path.dirname(__file__))

from preprocess import load_and_prepare, split_data


def train_and_evaluate(csv_path, output_dir='.'):
    print('Loading data...')
    df = load_and_prepare(csv_path)

    X_train, X_test, y_train, y_test = split_data(df)

    print('Vectorising text...')
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    print('Training Logistic Regression...')
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train_v, y_train)

    print('Evaluating...')
    y_pred = model.predict(X_test_v)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f'\nAccuracy : {acc:.4f} ({acc*100:.1f}%)')
    print(f'F1 Score : {f1:.4f}')

    print('\nFull classification report:')
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=['Extrovert', 'Introvert']
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Extrovert', 'Introvert'],
        yticklabels=['Extrovert', 'Introvert']
    )

    plt.title(
        f'Confusion Matrix (Accuracy {acc*100:.1f}%, F1 {f1:.2f})'
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()


    # CHANGED: Save confusion matrix inside images folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    cm_path = os.path.join(image_dir, 'confusion_matrix.png')


    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f'Confusion matrix saved to {cm_path}')


    # CHANGED: Ensure models folder exists
    os.makedirs(output_dir, exist_ok=True)


    # Save model
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    print('Model and vectorizer saved.')

    return acc, f1, cm



if __name__ == '__main__':
<<<<<<< HEAD

    # CHANGED: Use correct dataset and model folder paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    csv_path = os.path.join(
        base_dir,
        "data",
        "mbti_1.csv"
    )
    
    model_dir = os.path.join(
        base_dir,
        "models"
    )
    
    train_and_evaluate(csv_path, model_dir)
=======
    train_and_evaluate('../data/mbti_1.csv')
>>>>>>> e75f00c (Updated csv path)
