import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from utils import extract_features, create_word_features, create_character_features, create_pos_features, create_label_dictionary, convert_labels_to_indices
import ast

def load_csv_data(file_path: str) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Load data from CSV file and convert to sequences of words, POS tags, and labels
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (sequences, pos_tags, labels)
    """
    df = pd.read_csv(file_path)
    
    sequences = []
    pos_tags = []
    labels = []
    
    for _, row in df.iterrows():
        # Convert string representations of lists to actual lists
        sentence = row['Sentence'].split()
        pos = ast.literal_eval(row['POS'])
        tags = ast.literal_eval(row['Tag'])
        
        # Ensure lengths match
        if len(sentence) != len(tags) or len(sentence) != len(pos):
            print(f"Warning: Mismatched lengths in sentence: {len(sentence)} words, {len(pos)} POS tags, {len(tags)} NER tags")
            continue
            
        sequences.append(sentence)
        pos_tags.append(pos)
        labels.append(tags)
    
    return sequences, pos_tags, labels

def prepare_data(train_path: str, test_path: str, val_path: str = None) -> Dict:
    """
    Prepare data for CRF model training and evaluation
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        val_path: Optional path to validation CSV file
        
    Returns:
        Dictionary containing prepared data
    """
    # Load data
    train_sequences, train_pos, train_labels = load_csv_data(train_path)
    test_sequences, test_pos, test_labels = load_csv_data(test_path)
    print("Train data: Sentence")
    print(train_sequences[0])
    print("Train data: POS")
    print(train_pos[0])
    print("Train data: Labels")
    print(train_labels[0])


    # Create feature functions
    feature_functions = [create_word_features, create_character_features, create_pos_features]
    
    # Extract features
    train_features = [extract_features(seq, pos, feature_functions) 
                     for seq, pos in zip(train_sequences, train_pos)]
    test_features = [extract_features(seq, pos, feature_functions) 
                    for seq, pos in zip(test_sequences, test_pos)]
    print("Train data: Features")
    print(train_features[0])


    # Create label dictionary from training data
    label_dict = create_label_dictionary(train_labels)
    # Create label dictionary from all data to handle all possible labels
    all_labels = train_labels.copy()
    if val_path:
        val_sequences, val_pos, val_labels = load_csv_data(val_path)
        all_labels.extend(val_labels)
    
    label_dict = create_label_dictionary(all_labels)
    print("Label dictionary:")
    print(label_dict)
    
    # Convert labels to indices
    train_labels_indices = convert_labels_to_indices(train_labels, label_dict)
    test_labels_indices = convert_labels_to_indices(test_labels, label_dict)
    print("Train data: Labels indices")
    print(train_labels_indices[0])

    # Prepare validation data if provided
    val_data = None
    if val_path:
        val_features = [extract_features(seq, pos, feature_functions) 
                       for seq, pos in zip(val_sequences, val_pos)]
        val_labels_indices = convert_labels_to_indices(val_labels, label_dict)
        val_data = {
            'features': val_features,
            'labels': val_labels_indices,
            'sequences': val_sequences,
            'pos_tags': val_pos,
            'original_labels': val_labels
        }
    
    return {
        'train': {
            'features': train_features,
            'labels': train_labels_indices,
            'sequences': train_sequences,
            'pos_tags': train_pos,
            'original_labels': train_labels
        },
        'test': {
            'features': test_features,
            'labels': test_labels_indices,
            'sequences': test_sequences,
            'pos_tags': test_pos,
            'original_labels': test_labels
        },
        'validation': val_data,
        'label_dict': label_dict,
        'num_features': train_features[0].shape[1],
        'num_states': len(label_dict)
    }

def evaluate_predictions(predictions: List[np.ndarray], true_labels: List[np.ndarray], 
                        label_dict: Dict[str, int]) -> Dict[str, float]:
    """
    Evaluate model predictions using precision, recall, and F1 score
    
    Args:
        predictions: List of predicted label arrays
        true_labels: List of true label arrays
        label_dict: Dictionary mapping labels to indices
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Convert predictions and true labels to lists of strings
    pred_labels = []
    true_labels_str = []
    
    for pred, true in zip(predictions, true_labels):
        pred_labels.extend([list(label_dict.keys())[list(label_dict.values()).index(p)] for p in pred])
        true_labels_str.extend([list(label_dict.keys())[list(label_dict.values()).index(t)] for t in true])
    
    # Calculate metrics for each label
    metrics = {}
    for label in label_dict.keys():
        if label == 'O':  # Skip 'O' tag
            continue
            
        true_positives = sum(1 for p, t in zip(pred_labels, true_labels_str) 
                           if p == label and t == label)
        false_positives = sum(1 for p, t in zip(pred_labels, true_labels_str) 
                            if p == label and t != label)
        false_negatives = sum(1 for p, t in zip(pred_labels, true_labels_str) 
                            if p != label and t == label)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate overall metrics
    total_true_positives = sum(1 for p, t in zip(pred_labels, true_labels_str) if p == t and p != 'O')
    total_false_positives = sum(1 for p, t in zip(pred_labels, true_labels_str) if p != t and p != 'O')
    total_false_negatives = sum(1 for p, t in zip(pred_labels, true_labels_str) if p != t and t != 'O')
    
    overall_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
    overall_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1
    }
    
    return metrics 