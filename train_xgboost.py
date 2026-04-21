import pandas as pd
import numpy as np
from scipy.sparse import hstack
import os
from datetime import datetime

# Бібліотеки для Машинного Навчання
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb
import shap

# Бібліотеки для візуалізації
import matplotlib.pyplot as plt
import seaborn as sns

# Спроба завантажити бібліотеку стоп-слів (не є критичною, але бажаною)
try:
    from stop_words import get_stop_words
except ImportError:
    get_stop_words = None

# -- Налаштування

# Назва файлу з даними, який ми отримали на попередньому етапі (feature_extractor.py)
INPUT_CSV = "dataset_features.csv"

# Список статистичних ознак.
# !Цей список має точно співпадати з колонками у CSV файлі!
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

# -- Завантаження та перевірка даних

print(f"Завантаження даних з файлу: {INPUT_CSV}")

if not os.path.exists(INPUT_CSV):
    print(f"Помилка: Файл '{INPUT_CSV}' не знайдено.")
    print("Будь ласка, запустіть спочатку скрипт feature_extractor.py")
    exit()

# Зчитуємо CSV файл.
df = pd.read_csv(INPUT_CSV, sep=';')
print(f"Дані успішно завантажено. Кількість текстів: {len(df)}")

# Перевіряємо, чи всі необхідні стовбці присутні у файлі.
missing_cols = []
for col in STATISTICAL_FEATURES:
    if col not in df.columns:
        missing_cols.append(col)

if len(missing_cols) > 0:
    print(f"УВАГА! У файлі відсутні необхідні стовбці: {missing_cols}")
    exit()

# Коригування міток класів.
# Алгоритм XGBoost вимагає, щоб класи починалися з 0 (0, 1, 2, 3, 4).
# Якщо у нас класи 1-5, ми віднімаємо 1.
if df[TARGET_COLUMN].min() == 1:
    df[TARGET_COLUMN] = df[TARGET_COLUMN] - 1

y = df[TARGET_COLUMN] # Цільова змінна (клас складності)
X = df                # Вхідні дані (поки що весь датафрейм)

# Створюємо список назв класів для звітів (повертаємо до вигляду "Level 1"...)
target_names = [f"Клас {i+1}" for i in range(len(y.unique()))]
num_classes = len(target_names)

# -- Розділення на навчальну та тестову вибірки

# Використовуємо параметр stratify=y.
# Це гарантує, що в навчальній і тестовій вибірках буде однаковий відсоток текстів кожного класу.
# Для малих датасетів як в нашому випадку це важливо.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Кількість текстів для навчання: {len(X_train)}")
print(f"Кількість текстів для тестування: {len(X_test)}")

# -- Підготовка ознак. Feature Engineering.

# ЕТАП 1. Нормалізація статистичних ознак (Scaling)
# Різні ознаки мають різний масштаб. Наприклад: кількість слів ~500, а частка дієслів ~0.1.
# StandardScaler приводить їх до одного масштабу, щоб модель сприймала їх коректно.
print("Обробка статистичних ознак (Scaling)...")

scaler = StandardScaler()
# Навчаємо scaler тільки на тренувальних даних, щоб уникнути витоку даних
X_train_stat = scaler.fit_transform(X_train[STATISTICAL_FEATURES])
X_test_stat = scaler.transform(X_test[STATISTICAL_FEATURES])

# ЕТАП 2. Векторизація тексту (TF-IDF)
# Перетворюємо слова на цифри. Враховуємо окремі слова "unigrams" та пари слів "bigrams".
print("Обробка текстових ознак (TF-IDF)...")

# Спроба отримати українські стоп-слова (слова, які не несуть змісту: "і", "в", "на")
uk_stop_words = None
if get_stop_words:
    try:
        uk_stop_words = get_stop_words('ukrainian')
    except:
        pass

vectorizer = TfidfVectorizer(
    max_features=1500,     # Беремо тільки топ-1500 найважливіших слів
    stop_words=uk_stop_words,
    ngram_range=(1, 2)     # Уніграми та біграми
)

# Навчаємо векторизатор тільки на тренувальних текстах
X_train_tfidf = vectorizer.fit_transform(X_train[TEXT_COLUMN])
X_test_tfidf = vectorizer.transform(X_test[TEXT_COLUMN])

# ЕТАП 3. Об'єднання ознак (Stacking)
# Склеюємо матрицю слів TF-IDF та матрицю статистики в одну велику таблицю.
print("Об'єднання всіх ознак в єдиний масив...")
X_train_combined = hstack([X_train_tfidf, X_train_stat])
X_test_combined = hstack([X_test_tfidf, X_test_stat])

# Зберігаємо назви всіх ознак для подальшого аналізу (SHAP)
tfidf_names = vectorizer.get_feature_names_out()
# Об'єднуємо списки назв
all_feature_names = list(tfidf_names) + STATISTICAL_FEATURES

# -- Навчання моделі з підбором параметрів

print("\nПочаток автоматичного підбору параметрів (RandomizedSearchCV)...")
start_time = datetime.now()

# Сітка параметрів для перебору.
# RandomizedSearchCV буде випадковим чином вибирати комбінації з цього списку.
param_dist = {
    'n_estimators': [300, 500, 700],        # Кількість дерев
    'learning_rate': [0.01, 0.03, 0.05],    # Швидкість навчання
    'max_depth': [3, 4, 5],                 # Максимальна глибина дерева
    'reg_alpha': [1, 5, 10],                # L1: Відсікаємо зайві слова
    'reg_lambda': [1, 2, 5],                # L2: Згладжуємо ваги
    'min_child_weight': [3, 5, 7],          # Відсікаємо унікальні слова-викиди. Ігноруємо слова, що трапляються 1-2 рази
    'colsample_bytree': [0.4, 0.5, 0.6],    # Частка ознак для одного дерева. Беремо тільки 40-60% слів за раз
    'subsample': [0.7, 0.8]                 # Частка даних для одного дерева
}

# Базова модель XGBoost
base_model = xgb.XGBClassifier(
    objective='multi:softmax', # Мультикласова класифікація
    num_class=num_classes,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1 # Використовувати всі ядра процесора
)

# Налаштування пошуку
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=50,  # Кількість спроб (ітерацій)
    cv=5,       # Крос-валідація на 5 частин
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Запуск навчання
random_search.fit(X_train_combined, y_train)

duration = datetime.now() - start_time
clean_duration = str(duration).split('.')[0]

print(f"Навчання завершено за {clean_duration}")

best_model = random_search.best_estimator_
print(f"Найкращі знайдені параметри: {random_search.best_params_}")

# --Оцінка результатів

print("\nОцінка моделі на тестових даних...")
y_pred = best_model.predict(X_test_combined)

# Розрахунок загальної точності
acc = accuracy_score(y_test, y_pred)
acc_percent = round(acc * 100, 2)
acc_filename = str(acc_percent).replace('.', '_')
print(f"Точність (Accuracy): {acc_percent:.2f}%")

# Формуємо назву для файлів результатів
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
base_filename = f"XGB_Run_{timestamp}_Acc{acc_filename}"

# Створюємо папку для звітів
results_dir = "experiment_results_xgboost"
os.makedirs(results_dir, exist_ok=True)

# Збереження текстового звіту
report_text = classification_report(y_test, y_pred, target_names=target_names)
report_path = os.path.join(results_dir, f"{base_filename}_Report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Model: XGBoost + Hand-crafted Features + TF-IDF\n")
    f.write(f"Date: {timestamp}\n")
    f.write(f"Best Params: {random_search.best_params_}\n")
    f.write(f"Training completed in {clean_duration}\n")
    f.write(f"Final Accuracy: {acc_percent}%\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(report_text)

print(f"Звіт збережено у файл: {report_path}")

# Побудова матриці плутанини (Confusion Matrix)
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix (Accuracy: {acc_percent}%)")
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений клас')
plt.tight_layout()

cm_path = os.path.join(results_dir, f"{base_filename}_CM.png")
plt.savefig(cm_path)
plt.close()
print(f"Матрицю плутанини збережено: {cm_path}")

# -- Аналіз важливості ознак (SHAP)

print("\nРозрахунок SHAP (Інтерпретація моделі)...")
try:
    # SHAP працює повільно на великих даних, тому беремо вибірку (sample)
    # Перетворюємо в щільний формат (dense), бо TreeExplainer цього вимагає
    X_test_csr = X_test_combined.tocsr()
    samples_to_explain = 100 # Кількість прикладів для аналізу
    X_test_sample = X_test_csr[:samples_to_explain].toarray()
    
    # Ініціалізація пояснювача
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_sample)

    # Побудова графіка топ-20 ознак
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        X_test_sample, 
        feature_names=all_feature_names,
        class_names=target_names,
        max_display=20, 
        show=False
    )
    plt.title(f"SHAP Feature Importance (Top-20) - Accuracy: {acc_percent}%")
    plt.tight_layout()
    
    shap_path = os.path.join(results_dir, f"{base_filename}_SHAP.png")
    plt.savefig(shap_path)
    plt.close()
    print(f"SHAP графік збережено: {shap_path}")
    
except Exception as e:
    print(f"Не вдалося побудувати SHAP графік. Помилка: {e}")

print("\nПрограма завершила роботу успішно.")