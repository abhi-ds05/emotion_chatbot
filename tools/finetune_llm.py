import os
import pandas as pd
from datasets import Dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch

def load_and_prepare_dataset(csv_file):
    """
    Loads the MELD CSV dataset and encodes emotion labels as integers.
    
    Args:
        csv_file (str): Path to CSV file with columns 'Utterance' and 'Emotion'
    
    Returns:
        Dataset object from HuggingFace datasets library, label2id dict, id2label dict
    """
    df = pd.read_csv(csv_file)

    # Drop rows with missing values
    df = df.dropna(subset=['Utterance', 'Emotion'])

    # Encode emotion labels as integers
    unique_labels = sorted(df['Emotion'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df['labels'] = df['Emotion'].map(label2id)

    # Keep only necessary columns
    dataset = Dataset.from_pandas(df[['Utterance', 'labels']])
    dataset = dataset.rename_column("Utterance", "text")

    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])

    return dataset, label2id, id2label

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"
    OUTPUT_DIR = "./data/models/meld_emotion_model"
    LOGGING_DIR = "./logs"
    TRAIN_CSV = "./data/datasets/MELD/formatted_train_sent_emo.csv"
    VAL_CSV = "./data/datasets/MELD/formatted_dev_sent_emo.csv"
    NUM_EPOCHS = 3
    TRAIN_BATCH_SIZE = 16
    EVAL_BATCH_SIZE = 32
    SEED = 42

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset, label2id, id2label = load_and_prepare_dataset(TRAIN_CSV)
    val_dataset, _, _ = load_and_prepare_dataset(VAL_CSV)

    # Tokenize datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        seed=SEED,
        logging_dir=LOGGING_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model and tokenizer saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
