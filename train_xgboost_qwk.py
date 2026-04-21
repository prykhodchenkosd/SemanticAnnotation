import pandas as pd
import numpy as np
from scipy.sparse import hstack
import os
from datetime import datetime

# Бібліотеки для Машинного Навчання
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
import xgboost as xgb
import shap

# Бібліотеки для візуалізації
import matplotlib.pyplot as plt
import seaborn as sns

# Спроба завантажити бібліотеку стоп-слів
try:
    from stop_words import get_stop_words
except ImportError:
    get_stop_words = None


# -------------------------------
# Налаштування
# -------------------------------

INPUT_CSV = "dataset_features.csv"

STATISTICAL_FEATURES = [
    'n_sentences', 'n_words', 'n_syllables', 'n_complex_words',
    'ASL', 'ASW', 'AvgLetters',
    'n_noun', 'n_adj', 'n_verb', 'n_conj', 'n_prep', 'n_pron',
    'ratio_noun', 'ratio_verb', 'ratio_adj',
    'Flesch_RE', 'Flesch_Kincaid', 'Matskovskiy', 'Pisarek',
    'Solnyshkina'
]

TEXT_COLUMN = 'text'
TARGET_COLUMN = 'difficulty_level'


# -------------------------------
# Завантаження даних
# -------------------------------

print(f"Завантаження даних з файлу: {INPUT_CSV}")

if not os.path.exists(INPUT_CSV):
    print(f"Помилка: Файл '{INPUT_CSV}' не знайдено.")
    exit()

df = pd.read_csv(INPUT_CSV, sep=';')

print(f"Дані успішно завантажено. Кількість текстів: {len(df)}")


# Перевірка колонок
missing_cols = [c for c in STATISTICAL_FEATURES if c not in df.columns]

if missing_cols:
    print(f"УВАГА! У файлі відсутні стовбці: {missing_cols}")
    exit()


# -------------------------------
# Підготовка міток
# -------------------------------

if df[TARGET_COLUMN].min() == 1:
    df[TARGET_COLUMN] = df[TARGET_COLUMN] - 1

y = df[TARGET_COLUMN]
X = df

target_names = [f"Клас {i+1}" for i in range(len(y.unique()))]
num_classes = len(target_names)


# -------------------------------
# Train/Test split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Навчальна вибірка: {len(X_train)}")
print(f"Тестова вибірка: {len(X_test)}")


# -------------------------------
# Scaling статистичних ознак
# -------------------------------

print("Scaling статистичних ознак...")

scaler = StandardScaler()

X_train_stat = scaler.fit_transform(X_train[STATISTICAL_FEATURES])
X_test_stat = scaler.transform(X_test[STATISTICAL_FEATURES])


# -------------------------------
# TF-IDF
# -------------------------------

print("TF-IDF векторизація...")

uk_stop_words = None
if get_stop_words:
    try:
        uk_stop_words = get_stop_words('ukrainian')
    except:
        pass

vectorizer = TfidfVectorizer(
    max_features=1500,
    stop_words=uk_stop_words,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train[TEXT_COLUMN])
X_test_tfidf = vectorizer.transform(X_test[TEXT_COLUMN])


# -------------------------------
# Об'єднання ознак
# -------------------------------

print("Об'єднання ознак...")

X_train_combined = hstack([X_train_tfidf, X_train_stat])
X_test_combined = hstack([X_test_tfidf, X_test_stat])

tfidf_names = vectorizer.get_feature_names_out()
all_feature_names = list(tfidf_names) + STATISTICAL_FEATURES


# -------------------------------
# Підбір параметрів
# -------------------------------

print("\nПочаток RandomizedSearchCV...")
start_time = datetime.now()

param_dist = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 4, 5],
    'reg_alpha': [1, 5, 10],
    'reg_lambda': [1, 2, 5],
    'min_child_weight': [3, 5, 7],
    'colsample_bytree': [0.4, 0.5, 0.6],
    'subsample': [0.7, 0.8]
}

base_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train_combined, y_train)

duration = datetime.now() - start_time
clean_duration = str(duration).split('.')[0]

print(f"Навчання завершено за {clean_duration}")

best_model = random_search.best_estimator_
print(f"Найкращі параметри: {random_search.best_params_}")


# -------------------------------
# Оцінка моделі
# -------------------------------

print("\nОцінка на тестових даних...")

y_pred = best_model.predict(X_test_combined)

acc = accuracy_score(y_test, y_pred)
acc_percent = round(acc * 100, 2)

qwk = cohen_kappa_score(y_test, y_pred, weights='quadratic')
qwk_round = round(qwk, 4)

print(f"Accuracy: {acc_percent}%")
print(f"QWK (Quadratic Weighted Kappa): {qwk_round}")


# -------------------------------
# Підготовка файлів результатів
# -------------------------------

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
acc_filename = str(acc_percent).replace('.', '_')

base_filename = f"XGB_Run_{timestamp}_Acc{acc_filename}"

results_dir = "experiment_results_xgboost"
os.makedirs(results_dir, exist_ok=True)


# -------------------------------
# Звіт
# -------------------------------

report_text = classification_report(y_test, y_pred, target_names=target_names)

report_path = os.path.join(results_dir, f"{base_filename}_Report.txt")

with open(report_path, "w", encoding="utf-8") as f:

    f.write("Model: XGBoost + TF-IDF + Hand-crafted Features\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Training time: {clean_duration}\n")
    f.write(f"Best Params: {random_search.best_params_}\n\n")

    f.write(f"Accuracy: {acc_percent}%\n")
    f.write(f"Quadratic Weighted Kappa: {qwk_round}\n\n")

    f.write("Classification Report:\n")
    f.write(report_text)

print(f"Звіт збережено: {report_path}")


# -------------------------------
# Таблиця метрик
# -------------------------------

metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Quadratic Weighted Kappa"],
    "Value": [acc_percent, qwk_round]
})

table_path = os.path.join(results_dir, f"{base_filename}_metrics.csv")

metrics_table.to_csv(table_path, index=False)

print("\nТаблиця метрик:")
print(metrics_table)

print(f"Збережено: {table_path}")


# -------------------------------
# Confusion Matrix
# -------------------------------

plt.figure(figsize=(8,6))

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(f"Confusion Matrix (Acc: {acc_percent}% | QWK: {qwk_round})")
plt.ylabel("Справжній клас")
plt.xlabel("Передбачений клас")

plt.tight_layout()

cm_path = os.path.join(results_dir, f"{base_filename}_CM.png")

plt.savefig(cm_path)
plt.close()

print(f"Матрицю плутанини збережено: {cm_path}")


# -------------------------------
# SHAP
# -------------------------------

print("\nРозрахунок SHAP...")

try:

    X_test_csr = X_test_combined.tocsr()

    samples_to_explain = 100

    X_test_sample = X_test_csr[:samples_to_explain].toarray()

    explainer = shap.TreeExplainer(best_model)

    shap_values = explainer.shap_values(X_test_sample)

    plt.figure(figsize=(12,8))

    shap.summary_plot(
        shap_values,
        X_test_sample,
        feature_names=all_feature_names,
        class_names=target_names,
        max_display=20,
        show=False
    )

    plt.title(f"SHAP Feature Importance (Acc: {acc_percent}% | QWK: {qwk_round})")

    plt.tight_layout()

    shap_path = os.path.join(results_dir, f"{base_filename}_SHAP.png")

    plt.savefig(shap_path)

    plt.close()

    print(f"SHAP графік збережено: {shap_path}")

except Exception as e:

    print(f"SHAP не побудовано: {e}")


print("\nПрограма завершила роботу успішно.")