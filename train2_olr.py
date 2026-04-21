import os
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score

from mord import LogisticIT   # Ordinal Logistic Regression

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Налаштування
# ---------------------------

DATA_FILE = "dataset_features.csv"

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = f"ordinal_logistic_results_{timestamp}"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------
# Завантаження даних
# ---------------------------

print("Завантаження даних...")

df = pd.read_csv(DATA_FILE, sep=';')

# Перевірка міток
if df['difficulty_level'].min() == 1:
    df['difficulty_level'] = df['difficulty_level'] - 1

y = df['difficulty_level']

# Видаляємо нечислові колонки (текст, назви файлів тощо)
X = df.drop(columns=['text', 'difficulty_level'], errors='ignore')

# Залишаємо ТІЛЬКИ числові ознаки
X = X.select_dtypes(include=[np.number])

print("Кількість числових ознак:", X.shape[1])
print("Кількість текстів:", X.shape[0])
print("Типи даних після очищення:\n", X.dtypes)


# ---------------------------
# Train / Validation split
# ---------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ---------------------------
# Нормалізація ознак
# ---------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# ---------------------------
# Ordinal Logistic Regression
# ---------------------------

print("\nНавчання Ordinal Logistic Regression...")

model = LogisticIT()

model.fit(X_train, y_train)


# ---------------------------
# Прогноз
# ---------------------------

y_pred = model.predict(X_val)


# ---------------------------
# Метрики
# ---------------------------

accuracy = accuracy_score(y_val, y_pred)
qwk = cohen_kappa_score(y_val, y_pred, weights='quadratic')

print("\nAccuracy:", round(accuracy*100, 2), "%")
print("Quadratic Weighted Kappa:", round(qwk, 3))


# ---------------------------
# Classification Report
# ---------------------------

target_names = [f"Клас {i+1}" for i in range(len(np.unique(y_val)))]

report = classification_report(y_val, y_pred, target_names=target_names)

print("\nClassification Report:\n")
print(report)


# ---------------------------
# Збереження звіту
# ---------------------------

report_path = os.path.join(OUTPUT_DIR, "ordinal_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("Model: Ordinal Logistic Regression (Proportional Odds Model)\n\n")
    f.write(f"Accuracy: {round(accuracy*100,2)}%\n")
    f.write(f"Quadratic Weighted Kappa: {round(qwk,3)}\n\n")
    f.write(report)

print("Звіт збережено:", report_path)


# ---------------------------
# Confusion Matrix
# ---------------------------

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(f"Ordinal Logistic Regression\nAccuracy: {round(accuracy*100,2)}%")
plt.ylabel("Справжній клас")
plt.xlabel("Передбачений клас")

plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)

print("Матриця помилок збережена:", cm_path)