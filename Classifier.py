import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from datasets import DatasetDict
from transformers import get_scheduler

# Load SMS Spam dataset from Hugging Face
dataset = load_dataset("sms_spam")

# Split dataset into training and testing sets
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_data = dataset['train']
test_data = dataset['test']

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["sms"], padding="max_length", truncation=True, max_length=128)

# Apply tokenization
train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

# Convert labels to integers
train_data = train_data.rename_column("label", "labels")
test_data = test_data.rename_column("label", "labels")

# Remove unnecessary columns
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Load Pretrained BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Learning Rate Scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # 3 epochs
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training Loop
def train_model():
    model.train()
    for epoch in range(3):
        total_loss, total_correct = 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()
            total_correct += (outputs.logits.argmax(dim=-1) == batch["labels"]).sum().item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader):.4f}, Accuracy = {total_correct / len(train_data):.4f}")

# Evaluation Function
def evaluate_model():
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {acc:.4f}")

# HotFlip Adversarial Attack
def hotflip_attack(text):
    model.eval()
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get gradients of input tokens
    inputs["input_ids"].requires_grad = True
    outputs = model(**inputs)
    loss = criterion(outputs.logits, torch.tensor([1], dtype=torch.long, device=device))  # Targeted attack
    loss.backward()

    # Get most important token to flip
    grad = inputs["input_ids"].grad.abs().sum(dim=-1).squeeze()
    token_to_flip = grad.argmax().item()

    # Find a replacement token
    vocab = tokenizer.get_vocab()
    most_similar_token = None
    min_diff = float("inf")

    for token, idx in vocab.items():
        if idx == inputs["input_ids"][0][token_to_flip].item():
            continue  # Skip original token
        
        perturbed_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        perturbed_text[token_to_flip] = token  # Replace token
        perturbed_text = tokenizer.convert_tokens_to_string(perturbed_text)

        # Get new prediction
        perturbed_inputs = tokenizer(perturbed_text, return_tensors="pt").to(device)
        new_outputs = model(**perturbed_inputs)
        new_pred = new_outputs.logits.argmax(dim=-1).item()

        # Minimize loss difference
        new_loss = criterion(new_outputs.logits, torch.tensor([1], dtype=torch.long, device=device))
        diff = abs(new_loss.item() - loss.item())
        
        if diff < min_diff:
            min_diff = diff
            most_similar_token = token

    # Generate adversarial example
    adversarial_text = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    adversarial_text[token_to_flip] = most_similar_token  # Apply best perturbation
    adversarial_text = tokenizer.convert_tokens_to_string(adversarial_text)

    return adversarial_text

# Train and Evaluate
train_model()
evaluate_model()

# Example usage of HotFlip
original_text = "Congratulations! You have won a free iPhone. Click here to claim."
print("\nOriginal Text:", original_text)
adversarial_text = hotflip_attack(original_text)
print("Adversarial Text:", adversarial_text)
