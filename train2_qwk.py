import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)

from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Dataset ----------------

class ReadabilityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ---------------- Settings ----------------

MODEL_NAME = "youscan/ukr-roberta-base"
DATA_FILE = "dataset_features.csv"

MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 6
LEARNING_RATE = 2e-5
GRAD_ACCUM = 4

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = f"bert_qwk_results_{timestamp}"

# ---------------- Data ----------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, sep=';')

if df['difficulty_level'].min() == 1:
    df['difficulty_level'] -= 1

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["difficulty_level"].tolist(),
    test_size=0.2,
    stratify=df["difficulty_level"],
    random_state=42
)

print("Train:", len(train_texts))
print("Validation:", len(val_texts))

# -------- Class weights (важно для QWK)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

# ---------------- Tokenizer ----------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=False,
    max_length=MAX_LEN
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=False,
    max_length=MAX_LEN
)

train_dataset = ReadabilityDataset(train_encodings, train_labels)
val_dataset = ReadabilityDataset(val_encodings, val_labels)

data_collator = DataCollatorWithPadding(tokenizer)

# ---------------- Model ----------------

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5
)

# -------- Custom Trainer with class weights

class WeightedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")
        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(model.device)
        )

        loss = loss_fct(logits.view(-1, 5), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ---------------- Metrics ----------------

def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro"
    )

    qwk = cohen_kappa_score(labels, preds, weights="quadratic")

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "qwk": qwk
    }


# ---------------- Training ----------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LEARNING_RATE,

    warmup_ratio=0.1,

    weight_decay=0.01,

    fp16=torch.cuda.is_available(),

    evaluation_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="qwk",
    greater_is_better=True,

    logging_steps=20,

    report_to="none"
)

trainer = WeightedTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    tokenizer=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ---------------- Train ----------------

print("\nStart training...")

start_time = datetime.now()

trainer.train()

duration = datetime.now() - start_time

print("Training time:", str(duration).split('.')[0])

# ---------------- Evaluation ----------------

print("\nFinal evaluation...")

eval_result = trainer.evaluate()

acc = eval_result["eval_accuracy"]
qwk = eval_result["eval_qwk"]

print("Accuracy:", round(acc*100,2), "%")
print("Quadratic Weighted Kappa:", round(qwk,4))

# ---------------- Predictions ----------------

preds_output = trainer.predict(val_dataset)

y_preds = np.argmax(preds_output.predictions, axis=1)
y_true = preds_output.label_ids

target_names = [f"Клас {i+1}" for i in range(5)]

# ---------------- Report ----------------

report = classification_report(
    y_true,
    y_preds,
    target_names=target_names
)

report_path = os.path.join(OUTPUT_DIR, "final_report.txt")

with open(report_path, "w", encoding="utf-8") as f:

    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Accuracy: {round(acc*100,2)}%\n")
    f.write(f"Quadratic Weighted Kappa: {round(qwk,4)}\n\n")

    f.write(report)

print(report)

# ---------------- Confusion matrix ----------------

cm = confusion_matrix(y_true, y_preds)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap="Greens",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title(f"Confusion Matrix (QWK: {round(qwk,4)})")
plt.ylabel("True class")
plt.xlabel("Predicted class")

plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png"))

# ---------------- Save model ----------------

model_path = os.path.join(OUTPUT_DIR,"final_model")

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("Model saved to:", model_path)