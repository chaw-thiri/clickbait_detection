import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import logging
import os
import sys

# Import transformers explicitly
try:
    import transformers
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
except ImportError:
    print("Error: transformers module not found. Please install it using 'pip install transformers==4.52.4'")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log environment details
logger.info(f"Python version: {sys.version}")
logger.info(f"Transformers version: {transformers.__version__}")
logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Working directory: {os.getcwd()}")

# Load dataset
try:
    df = pd.read_csv("headlines_dataset.csv")
    df = df[df["label"].isin(["Clickbait", "Non-Clickbait"])]  # Filter out Unsafe for now
    df["label"] = df["label"].map({"Clickbait": 1, "Non-Clickbait": 0})
    logger.info(f"Loaded dataset with {len(df)} samples")
except FileNotFoundError:
    logger.error("headlines_dataset.csv not found. Ensure the file is in the same directory as this script.")
    raise
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Custom Dataset
class HeadlineDataset(Dataset):
    def __init__(self, headlines, labels, tokenizer, max_len=128):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        headline = str(self.headlines[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            headline,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and model
try:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    logger.info("Initialized tokenizer and model")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logger.info(f"Using device: {device}")

# Split dataset
try:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["headline"].values, df["label"].values, test_size=0.2, random_state=42
    )
    logger.info(f"Split dataset: {len(train_texts)} training, {len(val_texts)} validation")
except Exception as e:
    logger.error(f"Error splitting dataset: {e}")
    raise

# Create datasets and dataloaders
train_dataset = HeadlineDataset(train_texts, train_labels, tokenizer)
val_dataset = HeadlineDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Reduce LR by 10x each epoch
num_epochs = 5
best_val_loss = float("inf")
save_path = "./clickbait_detector_model"

# Training loop
try:
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

        scheduler.step()  # Update learning rate

    logger.info("Training completed")
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise

# Example inference
def predict_clickbait(headline):
    try:
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return "Clickbait" if probs[0][1] > probs[0][0] else "Non-Clickbait"
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return None

# Test
test_headlines = [
    "This celebrity's diet secret will SHOCK you!",
    "Engineers develop new shock-resistant material for earthquake-prone buildings."
]
for headline in test_headlines:
    result = predict_clickbait(headline)
    logger.info(f"Headline: {headline} -> Prediction: {result}")