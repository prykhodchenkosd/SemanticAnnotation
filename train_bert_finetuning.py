import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Бібліотеки для роботи з даними та метриками
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Бібліотеки Hugging Face Transformers (для нейромереж)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# Бібліотеки для графіків
import matplotlib.pyplot as plt
import seaborn as sns

# Клас Dataset
class ReadabilityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# -- Налаштування
MODEL_NAME = "youscan/ukr-roberta-base" # Спеціалізована українська модель youscan/ukr-roberta-base.
DATA_FILE = "dataset_features.csv" # Датасет(тексти)

# Гіперпараметри навчання
MAX_LEN = 512       # Максимальна довжина тексту
BATCH_SIZE = 4      # Кількість текстів за один крок
EPOCHS = 5          # Кількість епох
LEARNING_RATE = 2e-5 # Низька швидкість навчання

# Папка для збереження результатів
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = f"bert_finetune_results_{timestamp}"

# -- Підготовка даних

print("Завантаження та підготовка даних...")

if not os.path.exists(DATA_FILE):
    print(f"Помилка: Файл {DATA_FILE} не знайдено.")
    exit()

df = pd.read_csv(DATA_FILE, sep=';')

# Перевірка та коригування міток класів (мають бути 0..4)
if df['difficulty_level'].min() == 1:
    df['difficulty_level'] = df['difficulty_level'] - 1

# Розділення на тренувальну та валідаційну вибірки (Stratified)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['difficulty_level'].tolist(), 
    test_size=0.2, 
    stratify=df['difficulty_level'],
    random_state=42
)

print(f"Розмір тренувальної вибірки: {len(train_texts)}")
print(f"Розмір валідаційної вибірки: {len(val_texts)}")

# Токенізація

print(f"Завантаження токенізатора: {MODEL_NAME}")
# trust_remote_code=False для безпеки
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=False, use_safetensors=True)

# Перетворення тексту в числа
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)

train_dataset = ReadabilityDataset(train_encodings, train_labels)
val_dataset = ReadabilityDataset(val_encodings, val_labels)

# -- Ініціалізація моделі

print("Ініціалізація моделі...")

try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=5, # 5 класів складності
        use_safetensors=True,
        trust_remote_code=False
    )
except Exception as e:
    print(f"Помилка при завантаженні з use_safetensors=True: {e}")
    print("Спроба завантажити без use_safetensors...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=5,
        use_safetensors=False,
        trust_remote_code=False
    )

# Метод метрик
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# -- Налаштування тренування

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          
    num_train_epochs=EPOCHS,              
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,   
    warmup_steps=100,                
    weight_decay=0.01,               
    logging_dir=f'{OUTPUT_DIR}/logs',            
    logging_steps=10,
    eval_strategy="epoch",           
    save_strategy="epoch",           
    load_best_model_at_end=True,     
    metric_for_best_model="accuracy",
    learning_rate=LEARNING_RATE,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
)

# -- Початок Fine-Tuning

print("\nПочаток Fine-Tuning...")
start_time = datetime.now()

trainer.train()

duration = datetime.now() - start_time
clean_duration = str(duration).split('.')[0]
print(f"Навчання завершено за {clean_duration}")

# -- Оцінка

print("\nОцінка фінальної моделі...")
eval_result = trainer.evaluate()
acc_percent = round(eval_result['eval_accuracy']*100, 2)
print(f"Фінальна точність (Accuracy): {acc_percent}%")

preds_output = trainer.predict(val_dataset)
y_preds = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids
#target_names = [f"Level {i+1}" for i in range(5)]
target_names = [f"Клас {i+1}" for i in range(len(np.unique(y_true)) )]

# -- Збереження звітів

report = classification_report(y_true, y_preds, target_names=target_names)
report_path = os.path.join(OUTPUT_DIR, "final_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Model: Fine-Tuned BERT ({MODEL_NAME})\n")
    f.write(f"Training completed in {clean_duration}\n")
    f.write(f"Final Accuracy: {acc_percent}%\n\n")
    f.write(report)

print(f"Звіт збережено: {report_path}")
print(report)

# Матриця плутанини
cm = confusion_matrix(y_true, y_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title(f"Confusion Matrix (Acc: {acc_percent}%)")
plt.ylabel('Справжній клас')
plt.xlabel('Передбачений клас')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# Збереження моделі
model_save_path = os.path.join(OUTPUT_DIR, "final_model")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Модель збережено у: {model_save_path}")


#Добавь в скрипт и расчет qwk. И покажи qwk в итоговой таблице. дай полный итоговый скрипт с внесенными изменениями