import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

# -- Налаштування
INPUT_FILE = "dataset_features.csv" # Датасет з фічами та текстом
OUTPUT_FILE = "embeddings_roberta.csv" # Назва вихідного файлу
MODEL_NAME = "youscan/ukr-roberta-base" # Спеціалізована українська модель youscan/ukr-roberta-base.
BATCH_SIZE = 8 # Кількість текстів за один раз (зменшіть до 4 або 1, якщо мало RAM)

# -- Завантаження даних
print(f"Завантаження даних з {INPUT_FILE}...")
if not os.path.exists(INPUT_FILE):
    print("Помилка: Файл не знайдено.")
    exit()

df = pd.read_csv(INPUT_FILE, sep=';')

# Перевірка наявності тексту
if 'text' not in df.columns:
    print("Помилка: У CSV немає колонки 'text'.")
    exit()

texts = df['text'].tolist()
labels = df['difficulty_level'].tolist() # Зберігаємо мітки, щоб не загубити

print(f"Всього текстів для обробки: {len(texts)}")

# -- Завантаження моделі 
print(f"Завантаження моделі {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=False, use_safetensors=True)
except Exception as e:
    print(f"Помилка завантаження моделі: {e}")
    print("Спробуйте запустити: pip install transformers torch")
    exit()

# Переносимо на GPU, якщо (NVIDIA), Apple MPS, інакше CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Використовується NVIDIA GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Використовується Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("GPU не знайдено. Використовується CPU")

model.to(device)

# -- Генерація ембедінгів
# Ембедінг - це вектор (набір чисел), який описує зміст тексту.
# Для BERT-base це вектор розміром 768.

all_embeddings = []

print("Початок генерації ембедінгів. Це може зайняти певний час...")

# Проходимо по текстах батчами (групами)
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i : i + BATCH_SIZE]
    
    # Токенізація (перетворення слів у ID)
    # truncation=True обріже текст, якщо він довший за 512 токенів (обмеження BERT)
    inputs = tokenizer(
        batch_texts, 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad(): # Вимикаємо градієнти, бо ми не навчаємо, а тільки читаємо
        outputs = model(**inputs)
    
    # Отримуємо ембедінги.
    # Є кілька стратегій: взяти перший токен [CLS] або середнє всіх токенів (Mean Pooling).
    # Для RoBERTa часто використовують Mean Pooling.
    
    # last_hidden_state має розмір [batch, seq_len, 768]
    last_hidden_state = outputs.last_hidden_state
    
    # Attention mask показує, де реальні слова, а де пусті паддінги
    attention_mask = inputs['attention_mask']
    
    # Робимо Mean Pooling (ігноруючи паддінги)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    # Переносимо назад на CPU і додаємо в список
    all_embeddings.append(mean_embeddings.cpu().numpy())

# Об'єднуємо всі батчі в одну велику матрицю
final_embeddings = np.vstack(all_embeddings)
print(f"Генерацію завершено. Розмір матриці: {final_embeddings.shape}")

# -- Збереження результатів
# Створюємо DataFrame: Label + 768 колонок ембедінгів
emb_df = pd.DataFrame(final_embeddings)
# Додаємо префікс до колонок (emb_0, emb_1...)
emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

# Додаємо цільову мітку (рівень складності) на початок
emb_df.insert(0, "difficulty_level", labels)

# Зберігаємо
emb_df.to_csv(OUTPUT_FILE, index=False)
print(f"Дані збережено у: {OUTPUT_FILE}")
print("Тепер ви можете використовувати цей файл для навчання XGBoost.")