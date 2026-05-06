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

from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

import torch.nn as nn
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

NUM_CLASSES = 5
NUM_THRESHOLDS = NUM_CLASSES - 1

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = f"bert_coral_results_{timestamp}"


# ---------------- Data ----------------

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


# ---------------- CORAL Model ----------------

class CoralBERT(nn.Module):

    def __init__(self, model_name):

        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(hidden_size, NUM_THRESHOLDS)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.last_hidden_state[:,0]

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits


model = CoralBERT(MODEL_NAME)


# ---------------- CORAL utilities ----------------

def levels_from_labelbatch(labels):

    levels = []

    for label in labels:

        level = [1]*label + [0]*(NUM_THRESHOLDS-label)

        levels.append(level)

    return torch.tensor(levels, dtype=torch.float)


def coral_loss(logits, levels):

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        levels
    )

    return loss


def predict_classes(logits):

    prob = torch.sigmoid(logits)

    return torch.sum(prob > 0.5, dim=1)


# ---------------- Custom Trainer ----------------

class CoralTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")

        logits = model(**inputs)

        levels = levels_from_labelbatch(labels).to(logits.device)

        loss = coral_loss(logits, levels)

        return (loss, logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        labels = inputs.pop("labels")

        with torch.no_grad():

            logits = model(**inputs)

        preds = predict_classes(logits)

        return (None, preds.cpu().numpy(), labels.cpu().numpy())


# ---------------- Metrics ----------------

def compute_metrics(eval_pred):

    preds, labels = eval_pred

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


trainer = CoralTrainer(

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

trainer.train()


# ---------------- Evaluation ----------------

print("\nFinal evaluation...")

eval_result = trainer.evaluate()

acc = eval_result["eval_accuracy"]
qwk = eval_result["eval_qwk"]

print("Accuracy:", round(acc*100,2), "%")
print("Quadratic Weighted Kappa:", round(qwk,4))


# ---------------- Predictions ----------------

preds_output = trainer.predict(val_dataset)

y_preds = preds_output.predictions
y_true = preds_output.label_ids


target_names = [f"Клас {i+1}" for i in range(NUM_CLASSES)]

report = classification_report(
    y_true,
    y_preds,
    target_names=target_names
)

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
torch.save(model.state_dict(), os.path.join(model_path,"model.pt"))
tokenizer.save_pretrained(model_path)
print("Model saved to:", model_path)
