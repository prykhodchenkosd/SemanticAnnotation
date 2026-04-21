import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from datetime import datetime

# Бібліотеки для ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Бібліотеки для графіків
import matplotlib.pyplot as plt
import seaborn as sns

# -- Налаштування

# Файли з даними (мають бути в тій же папці)
EMBEDDINGS_FILE = "embeddings_roberta.csv" # Семантика (від BERT)
FEATURES_FILE = "dataset_features.csv"     # Статистика (від нашого екстрактора)

# Список статистичних ознак.
# ВАЖЛИВО: Додали 'Solnyshkina', бо вона є в новому feature_extractor.py
STAT_COLS = [
    'n_sentences', 'n_words', 'n_syllables', 'n_complex_words',
    'ASL', 'ASW', 'AvgLetters',
    'n_noun', 'n_adj', 'n_verb', 'n_conj', 'n_prep', 'n_pron',
    'ratio_noun', 'ratio_verb', 'ratio_adj',
    'Flesch_RE', 'Flesch_Kincaid', 'Matskovskiy', 'Pisarek',
    'Solnyshkina' 
]

# -- Завантаження даних

print("Завантаження вхідних файлів...")

if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(FEATURES_FILE):
    print(f"Помилка: Не знайдено файли {EMBEDDINGS_FILE} або {FEATURES_FILE}")
    print("Переконайтеся, що ви запустили feature_extractor.py ТА generate_embeddings.py")
    sys.exit(1)

# Завантажуємо обидва файли
df_emb = pd.read_csv(EMBEDDINGS_FILE)
df_stats = pd.read_csv(FEATURES_FILE, sep=';')

print(f"Завантажено ембедінгів: {len(df_emb)} рядків")
print(f"Завантажено статистики: {len(df_stats)} рядків")

# Перевірка: кількість рядків має співпадати ідеально
if len(df_emb) != len(df_stats):
    print("КРИТИЧНА ПОМИЛКА: Різна кількість текстів у файлах!")
    # Обрізаємо до мінімальної довжини, щоб код не впав, але це сигнал про проблему
    min_len = min(len(df_emb), len(df_stats))
    df_emb = df_emb.iloc[:min_len]
    df_stats = df_stats.iloc[:min_len]
else:
    print("Перевірка успішна: розмірності співпадають.")

# -- Формування гібридного датасету

print("Об'єднання ознак (Feature Fusion)...")

# 1. Формуємо Y (Правильні відповіді)
# Беремо з будь-якого файлу. Віднімаємо 1, щоб класи були 0,1,2,3,4 (вимога XGBoost)
y = df_emb['difficulty_level']
if y.min() == 1:
    y = y - 1

# 2. Формуємо частину X1 (Ембедінги BERT)
# Знаходимо всі колонки, що починаються на 'emb_' (emb_0, emb_1...)
emb_cols = [c for c in df_emb.columns if c.startswith('emb_')]
if not emb_cols:
    print("Помилка: У файлі ембедінгів не знайдено колонок 'emb_...'")
    sys.exit(1)
X_emb = df_emb[emb_cols]

# 3. Формуємо частину X2 (Статистика)
# Нормалізуємо статистику (StandardScaler), щоб великі числа (наприклад, 500 слів)
# не "заглушили" маленькі числа ембедінгів (0.05).
scaler = StandardScaler()
# Перевіряємо наявність колонок
missing = [c for c in STAT_COLS if c not in df_stats.columns]
if missing:
    print(f"Помилка: У файлі статистики немає колонок: {missing}")
    sys.exit(1)
    
X_stats_scaled = pd.DataFrame(
    scaler.fit_transform(df_stats[STAT_COLS]), 
    columns=STAT_COLS
)

# 4. СКЛЕЮВАННЯ (Concatenation)
# axis=1 означає "склеїти по горизонталі" (додати колонки справа)
X = pd.concat([X_emb, X_stats_scaled], axis=1)

print(f"Фінальний розмір вхідних даних (X): {X.shape}")
# Очікується: (750, 768 + 21) = (750, 789)

# -- Підготовка до навчання

# Розбиваємо на навчання та тест
# stratify=y гарантує, що в обох частинах буде порівну складних і легких текстів
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Створюємо імена класів для звітів "Level 1", "Level 2"...
target_names = [f"Клас {i+1}" for i in range(len(y.unique()))]
num_classes = len(target_names)

# -- Навчання моделі з підбором параметрів

print("\nПошук найкращих параметрів для Hybrid XGBoost...")
#start_time = time.time()
start_time = datetime.now()

# Сітка параметрів для перебору.
# RandomizedSearchCV буде випадковим чином вибирати комбінації з цього списку.
param_dist = {
    'n_estimators': [300, 500, 700],        # Кількість дерев
    'learning_rate': [0.01, 0.03, 0.05],    # Швидкість навчання
    'max_depth': [4, 5, 6],                 # Максимальна глибина дерева. Можна трохи глибше (5-6), бо ознаки "розумні"
    'reg_alpha': [0, 0.1, 1],               # L1: Відсікаємо зайві слова. М'якше, бо BERT дає корисні числа, а не нулі
    'reg_lambda': [1, 1.5, 2],              # L2: Згладжуємо ваги. Стандартно
    'min_child_weight': [1, 3, 5],          # Відсікаємо унікальні слова-викиди.
    'colsample_bytree': [0.6, 0.7, 0.8],    # Частка ознак для одного дерева. Можна брати більше ознак (60-80%)
    'subsample': [0.7, 0.8]                 # Частка даних для одного дерева.
}


xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', # Мультикласова класифікація
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1 # Використовувати всі ядра процесора
)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50, # Кількість спроб (ітерацій)
    cv=5,      # Крос-валідація на 5 частин
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

duration = datetime.now() - start_time
clean_duration = str(duration).split('.')[0]
print(f"Навчання завершено за {clean_duration}")

best_model = random_search.best_estimator_
print(f"Найкращі знайдені параметри: {random_search.best_params_}")

# -- Оцінка результатів

print("\nОцінка точності...")
y_pred = best_model.predict(X_test)

# Розрахунок точності
acc = accuracy_score(y_test, y_pred)
acc_percent = round(acc * 100, 2)
acc_filename = str(acc_percent).replace('.', '_')
print(f"Фінальна точність (Accuracy): {acc_percent}%")

# -- Збереження звітів

# Створюємо папку
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
base_filename = f"Hybrid_XGB_Run_{timestamp}_Acc{acc_filename}"
results_dir = "experiment_results_hybrid"
os.makedirs(results_dir, exist_ok=True)

# 1. Текстовий звіт
report = classification_report(y_test, y_pred, target_names=target_names)
with open(os.path.join(results_dir, f"{base_filename}_Report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Model: Hybrid (BERT Embeddings + Statistics)\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Best Params: {random_search.best_params_}\n")
    f.write(f"Training completed in {clean_duration}\n")
    f.write(f"Final Accuracy: {acc_percent}%\n\n")
    f.write("Detailed Report:\n")
    f.write(report)

print(f"Звіт збережено: {base_filename}_Report.txt")

# 2. Матриця плутанини
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix (Acc: {acc_percent}%)")
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений клас')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"{base_filename}_CM.png"))
plt.close()

# -- Важливість ознак

print("\nАналіз важливості ознак (Top-20)...")
# Для гібрида ми використовуємо вбудований метод XGBoost замість SHAP,
# тому що SHAP дуже повільний на 789 щільних ознаках.

importance = best_model.feature_importances_
feature_names = X.columns

# Сортуємо від найважливіших до найменш важливих
indices = np.argsort(importance)[::-1]
top_n = 20

plt.figure(figsize=(10, 8))
plt.title(f"Top-{top_n} Feature Importances (Hybrid Model)")
# Малюємо стовпчики
plt.barh(range(top_n), importance[indices[:top_n]], align="center")
# Підписуємо імена ознак
plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
plt.gca().invert_yaxis() # Щоб найважливіше було зверху
plt.tight_layout()

plt.savefig(os.path.join(results_dir, f"{base_filename}_Importance.png"))
plt.close()

print("Графік важливості збережено.")
print("\nРоботу завершено успішно!")

#Добавь в скрипт и расчет qwk. И покажи qwk в итоговой таблице. дай полный итоговый скрипт с внесенными изменениями