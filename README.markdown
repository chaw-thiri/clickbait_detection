# Clickbait Detector

A Python-based machine learning model to classify news headlines as either "Clickbait" or "Non-Clickbait" using DistilBERT, a lightweight transformer model.

## Project Overview

This project implements a binary text classification model to identify clickbait headlines. It uses the `DistilBertForSequenceClassification` model from the Hugging Face `transformers` library, fine-tuned on a dataset of labeled headlines (`headlines_dataset.csv`). The model is trained to distinguish between sensationalized clickbait and legitimate news headlines, achieving high accuracy and F1-score on validation data.

## Features

- **Dataset**: Processes `headlines_dataset.csv` with columns `headline` and `label` ("Clickbait" or "Non-Clickbait").
- **Model**: Fine-tunes DistilBERT for binary classification.
- **Training**: Uses AdamW optimizer with a step learning rate scheduler, training for 5 epochs.
- **Evaluation**: Reports training/validation loss, accuracy, and F1-score.
- **Inference**: Provides a `predict_clickbait` function to classify new headlines.
- **Logging**: Comprehensive logging for debugging and monitoring.
- **Model Saving**: Saves the best model based on validation loss.

## Requirements

- Python 3.7+
- Required packages (install via `pip`):
  ```bash
  pip install torch==2.0.1
  pip install transformers==4.52.4
  pip install pandas==1.5.3
  pip install scikit-learn==1.2.2
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd clickbait-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `headlines_dataset.csv` is in the project directory with the following format:
   ```csv
   headline,label
   "This celebrity's diet secret will SHOCK you!",Clickbait
   "Engineers develop new shock-resistant material.",Non-Clickbait
   ...
   ```

## Usage

1. **Train the Model**:
   Run the script to train the model:
   ```bash
   python clickbait_detector.py
   ```
   - The script loads the dataset, trains the model for 5 epochs, and saves the best model to `./clickbait_detector_model`.
   - Training progress, including loss, accuracy, and F1-score, is logged to the console.

2. **Inference**:
   Use the `predict_clickbait` function to classify new headlines. Example:
   ```python
   from clickbait_detector import predict_clickbait
   headline = "You won't believe what happened next!"
   result = predict_clickbait(headline)
   print(f"Headline: {headline} -> Prediction: {result}")
   ```

3. **Example Output**:
   ```plaintext
   Headline: This celebrity's diet secret will SHOCK you! -> Prediction: Clickbait
   Headline: Engineers develop new shock-resistant material for earthquake-prone buildings. -> Prediction: Non-Clickbait
   ```

## Project Structure

- `clickbait_detector.py`: Main script containing the model, dataset, training loop, and inference function.
- `headlines_dataset.csv`: Dataset file (not included; user must provide).
- `./clickbait_detector_model/`: Directory where the trained model and tokenizer are saved.
- `README.md`: This file.

## Notes

- **Dataset**: The model expects `headlines_dataset.csv` in the same directory. Ensure it contains valid headlines and labels.
- **Hardware**: Training uses CUDA if available, otherwise falls back to CPU.
- **Model Size**: DistilBERT is lightweight but still requires significant memory (~500MB for the model).
- **Error Handling**: The script includes robust error handling and logging for debugging.
- **Future Improvements**:
  - Support for additional labels (e.g., "Unsafe").
  - Hyperparameter tuning.
  - Web interface for real-time predictions.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/transformers).
- Inspired by the need to combat misleading online content.