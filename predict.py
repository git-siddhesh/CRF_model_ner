import pickle
import numpy as np
from data_loader import extract_features

def load_model_and_dict():
    """Load the trained CRF model and label dictionary"""
    with open('models/crf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    return model, label_dict

def predict_sequence(model, text, pos_tags):
    """
    Make predictions for a single sequence
    
    Args:
        model: Trained CRF model
        text: List of tokens
        pos_tags: List of POS tags
    
    Returns:
        List of predicted labels
    """
    # Extract features
    features = extract_features(text, pos_tags)
    
    # Make prediction
    pred_indices = model.predict(features)
    
    # Convert indices back to labels
    idx_to_label = {idx: label for label, idx in model.label_dict.items()}
    predictions = [idx_to_label[idx] for idx in pred_indices]
    
    return predictions

def main():
    # Load model and dictionary
    model, label_dict = load_model_and_dict()
    
    # Example usage
    text = ["John", "works", "at", "Google", "in", "New", "York"]
    pos_tags = ["NNP", "VBZ", "IN", "NNP", "IN", "NNP", "NNP"]
    
    predictions = predict_sequence(model, text, pos_tags)
    
    # Print results
    print("\nExample prediction:")
    print("Text:", " ".join(text))
    print("POS:", " ".join(pos_tags))
    print("Predictions:", " ".join(predictions))
    
    return model, label_dict

if __name__ == "__main__":
    main() 