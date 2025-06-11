from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config

def load_model_and_tokenizer(model_name_or_path, num_labels):
    """Loads a pre-trained model and tokenizer from Hugging Face or a local path."""
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(f"Tokenizer loaded successfully from {model_name_or_path}")

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels
        )
        print(f"Model loaded successfully from {model_name_or_path}")

        return model, tokenizer
    except OSError as e:
         print(f"Error loading model/tokenizer from {model_name_or_path}: {e}")
         print("Ensure the model name is correct or the path exists.")
         return None, None
    except Exception as e:
        print(f"An unexpected error occurred loading model/tokenizer: {e}")
        return None, None

if __name__ == '__main__':
    print(f"Attempting to load model: {config.MODEL_NAME}")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.NUM_LABELS)

    if model and tokenizer:
        print("Model and Tokenizer loaded successfully for testing.")
        sample_text = "This is a test sentence."
        encoding = tokenizer(sample_text, return_tensors='pt')
        print("Sample encoding:", encoding)
    else:
        print("Failed to load model and tokenizer for testing.")