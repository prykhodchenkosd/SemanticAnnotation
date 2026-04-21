import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from datetime import datetime

# Бібліотеки для ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

# Бібліотеки для графіків
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- НАСТРОЙКИ ---------------- #

EMBEDDINGS_FILE = "embeddings_roberta.csv"
FEATURES_FILE = "dataset_features.csv"

STAT_COLS = [
    'n_sentences','n_words','n_syllables','n_complex_words',
    'ASL','ASW','AvgLetters',
    'n_noun','n_adj','n_verb','n_conj','n_prep','n_pron',
    'ratio_noun','ratio_verb','ratio_adj',
    'Flesch_RE','Flesch_Kincaid','Matskovskiy','Pisarek',
    'Solnyshkina'
]


# ---------------- ЗАГРУЗКА ДАННЫХ ---------------- #

print("Завантаження вхідних файлів...")

if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(FEATURES_FILE):
    print(f"Помилка: Не знайдено файли {EMBEDDINGS_FILE} або {FEATURES_FILE}")
    sys.exit(1)

df_emb = pd.read_csv(EMBEDDINGS_FILE)
df_stats = pd.read_csv(FEATURES_FILE, sep=';')

print(f"Завантажено ембедінгів: {len(df_emb)} рядків")
print(f"Завантажено статистики: {len(df_stats)} рядків")

if len(df_emb) != len(df_stats):
    print("КРИТИЧНА ПОМИЛКА: Різна кількість текстів!")
    min_len = min(len(df_emb), len(df_stats))
    df_emb = df_emb.iloc[:min_len]
    df_stats = df_stats.iloc[:min_len]
else:
    print("Перевірка успішна: розмірності співпадають.")


# ---------------- ФОРМИРОВАНИЕ DATASET ---------------- #

print("Об'єднання ознак (Feature Fusion)...")

y = df_emb['difficulty_level']

if y.min() == 1:
    y = y - 1

emb_cols = [c for c in df_emb.columns if c.startswith('emb_')]

if not emb_cols:
    print("Помилка: не знайдено emb_* колонок")
    sys.exit(1)

X_emb = df_emb[emb_cols]

scaler = StandardScaler()

missing = [c for c in STAT_COLS if c not in df_stats.columns]

if missing:
    print(f"Помилка: немає колонок {missing}")
    sys.exit(1)

X_stats_scaled = pd.DataFrame(
    scaler.fit_transform(df_stats[STAT_COLS]),
    columns=STAT_COLS
)

X = pd.concat([X_emb, X_stats_scaled], axis=1)

print(f"Фінальний розмір X: {X.shape}")


# ---------------- TRAIN / TEST SPLIT ---------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

target_names = [f"Клас {i+1}" for i in range(len(y.unique()))]
num_classes = len(target_names)


# ---------------- HYPERPARAMETER SEARCH ---------------- #

print("\nПошук найкращих параметрів...")

start_time = datetime.now()

param_dist = {

    'n_estimators': [300,500,700],
    'learning_rate': [0.01,0.03,0.05],
    'max_depth': [4,5,6],
    'reg_alpha': [0,0.1,1],
    'reg_lambda': [1,1.5,2],
    'min_child_weight': [1,3,5],
    'colsample_bytree': [0.6,0.7,0.8],
    'subsample': [0.7,0.8]

}

xgb_model = xgb.XGBClassifier(

    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1

)

random_search = RandomizedSearchCV(

    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42

)

random_search.fit(X_train, y_train)

duration = datetime.now() - start_time
clean_duration = str(duration).split('.')[0]

print(f"Навчання завершено за {clean_duration}")

best_model = random_search.best_estimator_

print(f"Найкращі параметри: {random_search.best_params_}")


# ---------------- PREDICTIONS ---------------- #

print("\nОцінка точності...")

y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
acc_percent = round(acc * 100, 2)

# ----------- QWK -----------
qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

print(f"Accuracy: {acc_percent}%")
print(f"Quadratic Weighted Kappa (QWK): {round(qwk,4)}")


# ---------------- RESULTS DIRECTORY ---------------- #

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

acc_filename = str(acc_percent).replace('.', '_')

base_filename = f"Hybrid_XGB_Run_{timestamp}_Acc{acc_filename}"

results_dir = "experiment_results_hybrid"

os.makedirs(results_dir, exist_ok=True)


# ---------------- SUMMARY TABLE ---------------- #

summary_table = pd.DataFrame({
    "Metric":["Accuracy","QWK"],
    "Value":[round(acc,4), round(qwk,4)]
})

print("\nПідсумкова таблиця метрик:\n")
print(summary_table)


# ---------------- SAVE REPORT ---------------- #

report = classification_report(y_test, y_pred, target_names=target_names)

with open(os.path.join(results_dir,f"{base_filename}_Report.txt"),"w",encoding="utf-8") as f:

    f.write("Model: Hybrid (BERT Embeddings + Statistics)\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Best Params: {random_search.best_params_}\n")
    f.write(f"Training Time: {clean_duration}\n\n")

    f.write("Final Metrics:\n")
    f.write(f"Accuracy: {acc_percent}%\n")
    f.write(f"QWK: {round(qwk,4)}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

print("Звіт збережено.")


# ---------------- CONFUSION MATRIX ---------------- #

plt.figure(figsize=(8,6))

sns.heatmap(
    confusion_matrix(y_test,y_pred),
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(f"Confusion Matrix\nAcc:{acc_percent}%  QWK:{round(qwk,3)}")

plt.ylabel("True class")
plt.xlabel("Predicted class")

plt.tight_layout()

plt.savefig(os.path.join(results_dir,f"{base_filename}_CM.png"))

plt.close()


# ---------------- FEATURE IMPORTANCE ---------------- #

print("\nTop-20 важливих ознак")

importance = best_model.feature_importances_

feature_names = X.columns

indices = np.argsort(importance)[::-1]

top_n = 20

plt.figure(figsize=(10,8))

plt.title("Top Feature Importances")

plt.barh(range(top_n),importance[indices[:top_n]])

plt.yticks(range(top_n),[feature_names[i] for i in indices[:top_n]])

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(os.path.join(results_dir,f"{base_filename}_Importance.png"))

plt.close()

print("Графік важливості збережено.")

print("\nРоботу завершено успішно!")