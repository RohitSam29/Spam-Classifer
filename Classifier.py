import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
dataset = load_dataset("sms_spam")
texts = [d['sms'] for d in dataset['train']]
labels = [1 if d['label'] == 'spam' else 0 for d in dataset['train']]

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize texts
def tokenize(texts, labels):
    encoding = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    return encoding['input_ids'], encoding['attention_mask'], torch.tensor(labels)

train_input_ids, train_attention_mask, train_labels = tokenize(train_texts, train_labels)
test_input_ids, test_attention_mask, test_labels = tokenize(test_texts, test_labels)

# Dataset Class
class SpamDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Create datasets
train_dataset = SpamDataset(train_input_ids, train_attention_mask, train_labels)
test_dataset = SpamDataset(test_input_ids, test_attention_mask, test_labels)

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training function
def train(model, train_loader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

# Train model
train(model, train_loader, optimizer, criterion, epochs=1)

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Evaluate the model
evaluate(model, test_loader)

# Function to identify the most important word
def get_most_important_word(input_ids, attention_mask, label):
    input_ids = input_ids.unsqueeze(0).to(device)  
    attention_mask = attention_mask.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)

    # Get embeddings
    embeddings = model.get_input_embeddings()(input_ids)

    # Fix: Clone embeddings to make it a leaf variable and enable gradients
    embeddings = embeddings.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    loss = criterion(outputs.logits, label)

    # Compute gradients
    loss.backward()

    # Compute importance score for each token
    gradients = embeddings.grad.abs().sum(dim=-1).squeeze()
    important_idx = gradients.argmax().item()  # Find most important token index

    return important_idx

# Function to perform hotflip attack by replacing most important word
def hotflip_attack(text, model, tokenizer):
    encoding = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    input_ids = encoding['input_ids'].squeeze(0)
    attention_mask = encoding['attention_mask'].squeeze(0)

    # Dummy label (not used for actual classification)
    label = torch.tensor(0)

    # Get most important token index
    important_idx = get_most_important_word(input_ids, attention_mask, label)

    # Convert token IDs back to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    # Replace the most influential token with "car"
    tokens[important_idx] = "car"

    # Convert tokens back to text
    perturbed_text = tokenizer.convert_tokens_to_string(tokens)

    return perturbed_text

# Example
sample_text = "Congratulations! You have won a free lottery ticket."
print("Original:", sample_text)
print("Perturbed:", hotflip_attack(sample_text, model, tokenizer))
