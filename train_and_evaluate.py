import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, precision_recall_fscore_support
from CRF_Class import CRF, word_features
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and preprocess the dataset."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Debug: Print sample data and column names
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head()}")
    
    # Get column names based on your actual CSV structure
    # Modify these lines based on your CSV structure
    sentence_col = df.columns[0]  # Column with sentence ID
    word_col = df.columns[1]      # Column with words
    tag_col = df.columns[-1]      # Column with NER tags
    
    print(f"Using columns: Sentence={sentence_col}, Word={word_col}, Tag={tag_col}")
    
    # Group by sentence to create sequences
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    prev_sentence_id = None
    for _, row in df.iterrows():
        current_sentence_id = row[sentence_col]
        
        # Check if we're starting a new sentence
        if pd.isna(current_sentence_id):
            # End of file or missing value
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []
        elif prev_sentence_id is not None and current_sentence_id != prev_sentence_id:
            # New sentence detected
            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)
                current_sentence = []
                current_labels = []
        
        # Add the current word and tag if we have valid data
        if not pd.isna(current_sentence_id):
            current_sentence.append(str(row[word_col]))
            current_labels.append(str(row[tag_col]))
            prev_sentence_id = current_sentence_id
    
    # Add the last sentence if exists
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    
    return sentences, labels

def main():
    # Load training data
    print("Loading training data...")
    X_train, y_train = load_data('Train_CRF.csv')
    print(f"Loaded {len(X_train)} training sentences")
    
    # Split data into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    print(f"Split data: {len(X_train)} training sentences, {len(X_test)} test sentences")
    
    # Initialize and train the CRF model
    print("Initializing CRF model...")
    crf = CRF(feature_func=word_features, max_iterations=200)
    
    print("Training CRF model...")
    # Use a small subset for quick testing
    X_train_small = X_train[:100]
    y_train_small = y_train[:100]
    print(f"Training on small dataset: {len(X_train_small)} sentences")
    crf.fit(X_train_small, y_train_small)
    
    # After training, check if weights are non-zero
    print(f"Number of features learned: {crf.num_features}")
    print(f"Number of non-zero weights: {np.sum(crf.weights != 0)}")
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = crf.predict(X_test)
    
    # Flatten the sequences for evaluation
    y_test_flat = [label for seq in y_test for label in seq]
    y_pred_flat = [label for seq in y_pred for label in seq]
    
    # Make sure y_test_flat and y_pred_flat have the same length
    print(f"Test samples: {len(y_test_flat)}, Prediction samples: {len(y_pred_flat)}")

    # Check that we have actual predictions
    if len(y_test_flat) == len(y_pred_flat) and len(y_test_flat) > 0:
        # Calculate metrics safely
        accuracy = accuracy_score(y_test_flat, y_pred_flat)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_test_flat, y_pred_flat, average='weighted', zero_division=0
        )
        
        print("\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    else:
        print("Error: Mismatch in test and prediction lengths")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test_flat, y_pred_flat))

    # In your main() function, add this after loading data:
    print(f"Sample labels from training data: {y_train[0][:5]}")
    print(f"Unique labels in training data: {set([label for seq in y_train for label in seq])}")

    # After making predictions
    print("Sample predictions:")
    for i in range(min(3, len(y_test))):  # Show first 3 examples
        print(f"True: {y_test[i][:10]}")
        print(f"Pred: {y_pred[i][:10]}")
        print()

    # Check if predictions are valid
    unique_pred_labels = set([label for seq in y_pred for label in seq])
    print(f"Unique predicted labels: {unique_pred_labels}")

    # Check class distribution
    label_counts = Counter([label for seq in y_train for label in seq])
    print("Label distribution in training data:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main() 