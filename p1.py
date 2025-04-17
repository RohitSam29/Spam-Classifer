import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

#  Check for GPU (Faster Training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load Dataset (SMS Spam)
dataset = load_dataset("sms_spam")

#  Extract Texts and Labels
texts = [d["sms"] for d in dataset["train"]]
labels = [1 if d["label"] == "spam" else 0 for d in dataset["train"]]

#  Reduce Dataset for Faster Training
texts, labels = texts[:1000], labels[:1000]

#  Split Data into Train and Test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

#  Load Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

#  Tokenization Function
def tokenize_data(texts, labels):
    tokens = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    return tokens.input_ids, tokens.attention_mask, torch.tensor(labels)

#  Tokenize Train & Test Data
train_input_ids, train_attention_mask, train_labels = tokenize_data(train_texts, train_labels)
test_input_ids, test_attention_mask, test_labels = tokenize_data(test_texts, test_labels)

#  FIXED: Properly Define Dataset Class
class SpamDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

#  Create DataLoaders
batch_size = 16
train_dataset = SpamDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = SpamDataset(test_input_ids, test_attention_mask, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

#  Load DistilBERT Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

#  Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

#  Training Function
def train_model(model, train_loader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

#  Train Model (1 Epoch for Speed)
train_model(model, train_loader, optimizer, criterion, epochs=1)

#  Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

#  Evaluate Model
evaluate_model(model, test_loader)

#  HotFlip Attack (Adversarial Example)
def hotflip_attack(text):
    perturbed_text = list(text)
    if len(perturbed_text) > 3:
        perturbed_text[2] = 'X'  
    return "".join(perturbed_text)

#  Test HotFlip Attack
example_text = "Free entry in a weekly competition! Call now!"
print("Original Text: ", example_text)
print("Perturbed Text:", hotflip_attack(example_text))
