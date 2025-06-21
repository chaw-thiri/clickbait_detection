import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import logging
import os
import sys
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_PATH = "./clickbait_detector_model"

class HeadlineDataset(Dataset):
    """Custom Dataset for headlines."""
    def __init__(self, headlines, labels, tokenizer, max_length=128):
        self.headlines = headlines
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, idx):
        headline = str(self.headlines[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            headline,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_model_and_tokenizer():
    """Load pre-trained model and tokenizer from MODEL_PATH."""
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model directory '{MODEL_PATH}' not found. Training required.")
            return None, None, None
        
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info("Loaded pre-trained model and tokenizer")
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None

def predict_clickbait(headline):
    """
    Predict if a headline is Clickbait or Non-Clickbait using the pre-trained model.
    
    Args:
        headline (str): The headline to classify
    
    Returns:
        tuple: (label, confidence) or (None, None) if prediction fails
    """
    try:
        tokenizer, model, device = load_model_and_tokenizer()
        if model is None:
            logger.error("No model available. Please train the model first.")
            return None, None
        
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        label = "Clickbait" if predicted_class == 1 else "Non-Clickbait"
        logger.info(f"Headline: {headline} -> Prediction: {label} (Confidence: {confidence:.2%})")
        return label, confidence
    except Exception as e:
        logger.error(f"Error predicting headline: {e}")
        return None, None

def train_model(force_training=False, dataset_file="headlines_dataset.csv"):
    """
    Train the DistilBERT model if needed.
    
    Args:
        force_training (bool): Force retraining even if model exists
        dataset_file (str): Path to dataset CSV
    
    Returns:
        bool: True if training successful or model exists, False otherwise
    """
    try:
        # Check if model exists
        if os.path.exists(MODEL_PATH) and not force_training:
            logger.info(f"Model found at {MODEL_PATH}. Skipping training.")
            return True
        
        # Load dataset
        try:
            df = pd.read_csv(dataset_file)
            if df.empty or "headline" not in df.columns or "label" not in df.columns:
                raise ValueError("Invalid or empty dataset")
        except FileNotFoundError:
            logger.error(f"Dataset {dataset_file} not found")
            return False
        
        logger.info(f"Loaded dataset with {len(df)} samples")
        
        # Filter out invalid labels
        valid_labels = ["Clickbait", "Non-Clickbait"]
        df = df[df["label"].isin(valid_labels)].copy()
        if df.empty:
            logger.error("No valid labels in dataset")
            return False
        
        # Encode labels
        df["label"] = df["label"].map({"Clickbait": 1, "Non-Clickbait": 0})
        
        # Split dataset
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        logger.info(f"Split dataset: {len(train_df)} training, {len(val_df)} validation")
        
        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Prepare datasets
        train_dataset = HeadlineDataset(
            train_df["headline"].values,
            train_df["label"].values,
            tokenizer
        )
        val_dataset = HeadlineDataset(
            val_df["headline"].values,
            val_df["label"].values,
            tokenizer
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        num_epochs = 5
        best_val_loss = float("inf")
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
            
            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(batch["labels"].cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = accuracy_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(MODEL_PATH, exist_ok=True)
                model.save_pretrained(MODEL_PATH)
                tokenizer.save_pretrained(MODEL_PATH)
                logger.info(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
        
        logger.info("Training completed")
        return True
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

if __name__ == "__main__":
    # Train model if needed
    train_model()
    # Test predictions
    test_headlines = [
        "This celebrity's diet secret will SHOCK you!",
        "Engineers develop new shock-resistant material for earthquake-prone buildings."
    ]
    for headline in test_headlines:
        label, confidence = predict_clickbait(headline)
        if label:
            print(f"Headline: {headline} -> {label} (Confidence: {confidence:.2%})")
        else:
            print(f"Headline: {headline} -> Prediction failed")