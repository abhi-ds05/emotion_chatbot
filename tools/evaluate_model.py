# tools/evaluate_model.py

import pandas as pd
import torch
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

def load_test_dataset(csv_file, label2id):
    """
    Load and preprocess test dataset from CSV.
    """
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['Utterance', 'Emotion'])
    df['labels'] = df['Emotion'].map(label2id)
    dataset = Dataset.from_pandas(df[['Utterance', 'labels']])
    dataset = dataset.rename_column("Utterance", "text")
    dataset = dataset.remove_columns(["__index_level_0__"] if "__index_level_0__" in dataset.column_names else [])
    return dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding=True)

def evaluate(model_dir, test_csv):
    """
    Evaluate the fine-tuned model on the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Obtain id2label mapping from model config
    id2label = model.config.id2label
    label2id = {label: int(idx) for idx, label in id2label.items()}

    # Load test dataset
    test_dataset = load_test_dataset(test_csv, label2id=label2id)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare dataloader
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)

    metric = load_metric("accuracy")

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch['labels'].cpu().tolist())

    accuracy = metric.compute(predictions=all_preds, references=all_labels)['accuracy']
    print(f"Accuracy on test set: {accuracy:.4f}")

if __name__ == "__main__":
    MODEL_DIR = "./data/models/meld_emotion_model"   # path to your fine-tuned model directory
    TEST_CSV = "./data/datasets/MELD/formatted_test_sent_emo.csv"  # path to your formatted test CSV

    evaluate(MODEL_DIR, TEST_CSV)
