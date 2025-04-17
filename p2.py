import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.nn.functional import cross_entropy

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset (Huggingface SMS Spam Dataset)
dataset = load_dataset("sms_spam")

# Extract texts and labels
texts = [d['sms'] for d in dataset['train']]
labels = [1 if d['label'] == 'spam' else 0 for d in dataset['train']]

# Split into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
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

# Train model (1 epoch for demonstration)
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

# Hotflip Gradient-Based Attack
def hotflip_gradient_attack(model, tokenizer, text, label):
    model.eval()
    
    # Tokenize input
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    label = torch.tensor([label]).to(device)

    # Get input embeddings
    input_embeddings = model.get_input_embeddings()(input_ids)
    input_embeddings.retain_grad()

    # Forward pass
    outputs = model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
    loss = cross_entropy(outputs.logits, label)
    loss.backward()

    # Gradients of embeddings
    grads = input_embeddings.grad  # Shape: (1, seq_len, embed_dim)

    # Find the most sensitive token (highest gradient norm)
    grad_norms = grads.norm(dim=2).squeeze()  # Norm along embedding dimension
    most_sensitive_idx = grad_norms.argmax().item()

    # Get the original tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

    print(f"Most sensitive token: '{tokens[most_sensitive_idx]}' at position {most_sensitive_idx}")

    # For demonstration, replace with 'UNK'
    tokens[most_sensitive_idx] = '[UNK]'

    # Reconstruct perturbed text
    perturbed_text = tokenizer.convert_tokens_to_string(tokens)
    print(f"Original Text: {text}")
    print(f"Perturbed Text: {perturbed_text}")

    return perturbed_text

# Example usage of Hotflip attack
sample_text = "Congratulations! Rohit you won the match."
sample_label = 1  # Spam
perturbed_text = hotflip_gradient_attack(model, tokenizer, sample_text, sample_label)
