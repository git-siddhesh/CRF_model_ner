import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import ast
from sklearn.feature_selection import mutual_info_classif

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

def get_word_shape(word: str) -> str:
    """Get word shape (e.g., 'John' -> 'Xxxx')"""
    shape = ''
    for char in word:
        if char.isupper():
            shape += 'X'
        elif char.islower():
            shape += 'x'
        elif char.isdigit():
            shape += 'd'
        else:
            shape += char
    return shape

def get_word_pattern(word: str) -> str:
    """Get word pattern (e.g., 'John-2' -> 'Aa-N')"""
    pattern = ''
    for i, char in enumerate(word):
        if char.isupper():
            pattern += 'A' if i == 0 else 'a'
        elif char.islower():
            pattern += 'a'
        elif char.isdigit():
            pattern += 'N'
        else:
            pattern += char
    return pattern

def apply_feature_template(template: Dict, sequence: List[str], pos_tags: List[str], position: int) -> List[float]:
    """
    Apply a feature template to extract features
    """
    features = []
    seq_len = len(sequence)
    offset = template['offset']
    target_pos = position + offset
    
    # Skip if position is out of bounds
    if target_pos < 0 or target_pos >= seq_len:
        if template['type'] in ['word', 'pos']:
            return [0.0]  # Single feature for OOV
        return [0.0] * 4  # Standard number of features for other types
    
    word = sequence[target_pos]
    
    if template['type'] == 'word':
        features.append(1.0)  # Presence of word
        
    elif template['type'] == 'prefix':
        length = template['length']
        prefix = word[:length] if len(word) >= length else word
        features.extend([
            1.0 if prefix.isupper() else 0.0,
            1.0 if prefix.istitle() else 0.0,
            len(prefix),
            1.0 if any(c.isdigit() for c in prefix) else 0.0
        ])
        
    elif template['type'] == 'suffix':
        length = template['length']
        suffix = word[-length:] if len(word) >= length else word
        features.extend([
            1.0 if suffix.isupper() else 0.0,
            1.0 if suffix.islower() else 0.0,
            len(suffix),
            1.0 if any(c.isdigit() for c in suffix) else 0.0
        ])
        
    elif template['type'] == 'is_capitalized':
        features.append(1.0 if word.istitle() else 0.0)
        
    elif template['type'] == 'has_number':
        features.append(1.0 if any(c.isdigit() for c in word) else 0.0)
        
    elif template['type'] == 'has_hyphen':
        features.append(1.0 if '-' in word else 0.0)
        
    elif template['type'] == 'pos':
        pos = pos_tags[target_pos]
        features.extend([
            1.0 if pos.startswith(('NN', 'NNP', 'NNPS', 'NNS')) else 0.0,  # noun
            1.0 if pos.startswith('VB') else 0.0,  # verb
            1.0 if pos.startswith('JJ') else 0.0,  # adjective
            1.0 if pos.startswith('RB') else 0.0   # adverb
        ])
        
    elif template['type'] == 'pos_bigram':
        if target_pos + 1 < seq_len:
            pos1 = pos_tags[target_pos]
            pos2 = pos_tags[target_pos + 1]
            features.extend([
                1.0 if pos1.startswith('NN') and pos2.startswith('VB') else 0.0,  # noun-verb
                1.0 if pos1.startswith('JJ') and pos2.startswith('NN') else 0.0,  # adj-noun
                1.0 if pos1.startswith('RB') and pos2.startswith('VB') else 0.0,  # adv-verb
                1.0 if pos1.startswith('DT') and pos2.startswith('NN') else 0.0   # det-noun
            ])
        else:
            features.extend([0.0] * 4)
            
    elif template['type'] == 'word_shape':
        shape = get_word_shape(word)
        features.extend([
            1.0 if shape[0] == 'X' else 0.0,  # Starts with uppercase
            1.0 if all(c == 'x' for c in shape[1:]) else 0.0,  # Rest lowercase
            1.0 if any(c == 'd' for c in shape) else 0.0,  # Contains digit
            len(shape)  # Length of shape
        ])
        
    elif template['type'] == 'word_pattern':
        pattern = get_word_pattern(word)
        features.extend([
            1.0 if pattern.startswith('A') else 0.0,  # Proper case
            1.0 if all(c == 'a' for c in pattern[1:]) else 0.0,  # Rest lowercase
            1.0 if 'N' in pattern else 0.0,  # Contains number
            1.0 if '-' in pattern else 0.0  # Contains hyphen
        ])
    
    return features

def extract_features(sequence: List[str], pos_tags: List[str]) -> np.ndarray:
    """
    Extract features from a sequence using templates
    """
    # Get feature templates from CRF model
    from crf_model import CRF
    dummy_crf = CRF(2, 2)  # Temporary instance to get templates
    templates = dummy_crf.feature_templates
    
    features = []
    for i in range(len(sequence)):
        position_features = []
        for template in templates:
            template_features = apply_feature_template(template, sequence, pos_tags, i)
            position_features.extend(template_features)
        features.append(position_features)
    
    return np.array(features)

def create_word_features(sequence: List[str], position: int) -> List[float]:
    """
    Create enhanced word-level features
    """
    features = []
    word = sequence[position]
    
    # Current word features
    features.extend([
        1.0 if word.isupper() else 0.0,  # all uppercase
        1.0 if word.istitle() else 0.0,  # capitalized
        1.0 if word.islower() else 0.0,  # all lowercase
        1.0 if word.isdigit() else 0.0,  # numeric
        1.0 if any(c.isdigit() for c in word) else 0.0,  # contains number
        1.0 if '-' in word else 0.0,  # contains hyphen
        1.0 if '.' in word else 0.0,  # contains period
        len(word),  # word length
        1.0 if position == 0 else 0.0,  # start of sentence
        1.0 if position == len(sequence) - 1 else 0.0,  # end of sentence
    ])
    
    # Previous word features (window size 2)
    for i in range(2):
        prev_pos = position - (i + 1)
        if prev_pos >= 0:
            prev_word = sequence[prev_pos]
            features.extend([
                1.0 if prev_word.isupper() else 0.0,
                1.0 if prev_word.istitle() else 0.0,
                1.0 if prev_word.isdigit() else 0.0,
                len(prev_word)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Next word features (window size 2)
    for i in range(2):
        next_pos = position + (i + 1)
        if next_pos < len(sequence):
            next_word = sequence[next_pos]
            features.extend([
                1.0 if next_word.isupper() else 0.0,
                1.0 if next_word.istitle() else 0.0,
                1.0 if next_word.isdigit() else 0.0,
                len(next_word)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    return features

def create_character_features(sequence: List[str], position: int) -> List[float]:
    """
    Create enhanced character-level features
    """
    features = []
    word = sequence[position]
    
    # Character type features
    features.extend([
        sum(c.isupper() for c in word) / max(len(word), 1),  # proportion of uppercase
        sum(c.isdigit() for c in word) / max(len(word), 1),  # proportion of digits
        sum(c in '.,;:!?' for c in word) / max(len(word), 1),  # proportion of punctuation
    ])
    
    # Prefix and suffix features (up to length 4)
    for n in range(1, 5):
        if len(word) >= n:
            prefix = word[:n]
            suffix = word[-n:]
            features.extend([
                1.0 if prefix.isupper() else 0.0,
                1.0 if suffix.isupper() else 0.0,
                1.0 if prefix.isdigit() else 0.0,
                1.0 if suffix.isdigit() else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Special character features
    features.extend([
        1.0 if any(c.isupper() for c in word) else 0.0,  # contains uppercase
        1.0 if any(c.isdigit() for c in word) else 0.0,  # contains digit
        1.0 if any(not c.isalnum() for c in word) else 0.0,  # contains special char
    ])
    
    return features

def create_pos_features(pos_tags: List[str], position: int) -> List[float]:
    """
    Create enhanced POS tag features
    """
    features = []
    pos = pos_tags[position]
    
    # Current POS features
    pos_categories = {
        'noun': pos.startswith(('NN', 'NNP', 'NNPS', 'NNS')),
        'verb': pos.startswith(('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')),
        'adj': pos.startswith(('JJ', 'JJR', 'JJS')),
        'adv': pos.startswith(('RB', 'RBR', 'RBS')),
        'det': pos.startswith(('DT', 'WDT')),
        'prep': pos.startswith('IN'),
        'pron': pos.startswith(('PRP', 'WP')),
        'num': pos.startswith('CD'),
        'proper': pos.startswith('NNP'),
        'foreign': pos.startswith('FW'),
        'symbol': pos.startswith('SYM'),
        'modal': pos.startswith('MD'),
        'wh': pos.startswith('W'),
        'conj': pos.startswith(('CC', 'IN')),
        'particle': pos.startswith('RP')
    }
    
    features.extend([1.0 if v else 0.0 for v in pos_categories.values()])
    
    # Previous POS features (window size 2)
    for i in range(2):
        prev_pos = position - (i + 1)
        if prev_pos >= 0:
            prev_tag = pos_tags[prev_pos]
            features.extend([
                1.0 if prev_tag.startswith(('NN', 'NNP', 'NNPS', 'NNS')) else 0.0,  # noun
                1.0 if prev_tag.startswith('NNP') else 0.0,  # proper noun
                1.0 if prev_tag.startswith(('JJ', 'JJR', 'JJS')) else 0.0,  # adjective
                1.0 if prev_tag.startswith('IN') else 0.0,  # preposition
                1.0 if prev_tag.startswith(('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')) else 0.0  # verb
            ])
        else:
            features.extend([0.0] * 5)
    
    # Next POS features (window size 2)
    for i in range(2):
        next_pos = position + (i + 1)
        if next_pos < len(pos_tags):
            next_tag = pos_tags[next_pos]
            features.extend([
                1.0 if next_tag.startswith(('NN', 'NNP', 'NNPS', 'NNS')) else 0.0,  # noun
                1.0 if next_tag.startswith('NNP') else 0.0,  # proper noun
                1.0 if next_tag.startswith(('JJ', 'JJR', 'JJS')) else 0.0,  # adjective
                1.0 if next_tag.startswith('IN') else 0.0,  # preposition
                1.0 if next_tag.startswith(('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')) else 0.0  # verb
            ])
        else:
            features.extend([0.0] * 5)
    
    return features

def create_label_dictionary(labels_list: List[List[str]]) -> Dict[str, int]:
    """
    Create a dictionary mapping labels to indices with BIO scheme handling
    """
    # Ensure 'O' tag is at index 0
    label_set = {'O'}
    
    # Add B- and I- tags
    for labels in labels_list:
        for label in labels:
            if label != 'O':
                # Add both B- and I- versions of each entity type
                entity_type = label[2:]  # Remove B- or I-
                label_set.add(f'B-{entity_type}')
                label_set.add(f'I-{entity_type}')
    
    # Sort labels to ensure consistent ordering
    sorted_labels = sorted(list(label_set))
    if 'O' in sorted_labels:
        sorted_labels.remove('O')
        sorted_labels = ['O'] + sorted_labels
    
    return {label: idx for idx, label in enumerate(sorted_labels)}

def convert_labels_to_indices(labels_list: List[List[str]], label_dict: Dict[str, int]) -> List[np.ndarray]:
    """
    Convert string labels to indices
    """
    return [np.array([label_dict[label] for label in labels]) for labels in labels_list]

def select_informative_features(features_list: List[np.ndarray], labels_list: List[np.ndarray], 
                              threshold: float = 0.001) -> np.ndarray:
    """
    Select informative features using mutual information
    """
    # Flatten features and labels for MI calculation
    flat_features = np.vstack(features_list)
    flat_labels = np.concatenate([labels for labels in labels_list])
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(flat_features, flat_labels)
    
    # Select features above threshold
    selected_features = mi_scores > threshold
    
    print(f"Selected {np.sum(selected_features)} features out of {len(mi_scores)}")
    return selected_features

def apply_feature_selection(features_list: List[np.ndarray], selected_features: np.ndarray) -> List[np.ndarray]:
    """
    Apply feature selection mask to features
    """
    return [features[:, selected_features] for features in features_list]

def prepare_data(train_path: str, test_path: str, val_path: str = None, val_split: float = 0.1) -> Dict:
    """
    Prepare data for CRF model training and evaluation with feature selection
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        val_path: Path to validation data (optional)
        val_split: Fraction of training data to use for validation if val_path is None
    """
    # Load all data first
    train_sequences, train_pos, train_labels = load_csv_data(train_path)
    test_sequences, test_pos, test_labels = load_csv_data(test_path)
    
    # Collect all labels to create a complete label dictionary
    all_labels = train_labels.copy()
    all_labels.extend(test_labels)
    
    if val_path:
        val_sequences, val_pos, val_labels = load_csv_data(val_path)
        all_labels.extend(val_labels)
    else:
        # Split training data into train and validation sets
        num_val = int(len(train_sequences) * val_split)
        indices = np.random.permutation(len(train_sequences))
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        # Create validation set
        val_sequences = [train_sequences[i] for i in val_indices]
        val_pos = [train_pos[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]
        
        # Update training set
        train_sequences = [train_sequences[i] for i in train_indices]
        train_pos = [train_pos[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        
        print(f"Split training data: {len(train_sequences)} train, {len(val_sequences)} validation sequences")
    
    # Create label dictionary from all available data
    label_dict = create_label_dictionary(all_labels)
    print("Label dictionary:", label_dict)
    
    # Extract initial features
    train_features = [extract_features(seq, pos) for seq, pos in zip(train_sequences, train_pos)]
    test_features = [extract_features(seq, pos) for seq, pos in zip(test_sequences, test_pos)]
    
    # Convert labels to indices
    train_labels_indices = convert_labels_to_indices(train_labels, label_dict)
    test_labels_indices = convert_labels_to_indices(test_labels, label_dict)
    
    # Perform feature selection on training data
    selected_features = select_informative_features(train_features, train_labels_indices)
    
    # Apply feature selection
    train_features = apply_feature_selection(train_features, selected_features)
    test_features = apply_feature_selection(test_features, selected_features)
    
    # Prepare validation data if provided
    val_data = None
    if val_path:
        val_features = [extract_features(seq, pos) for seq, pos in zip(val_sequences, val_pos)]
        val_features = apply_feature_selection(val_features, selected_features)
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
        'num_states': len(label_dict),
        'selected_features': selected_features
    }

def evaluate_predictions(predictions: List[np.ndarray], true_labels: List[np.ndarray], 
                        label_dict: Dict[str, int]) -> Dict[str, Dict[str, float]]:
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
    
    # Create reverse mapping from index to label
    idx_to_label = {idx: label for label, idx in label_dict.items()}
    
    # Convert all labels to strings
    for pred, true in zip(predictions, true_labels):
        pred_labels.extend([idx_to_label[p] for p in pred])
        true_labels_str.extend([idx_to_label[t] for t in true])
    
    # Initialize metrics dictionary
    metrics = {
        'per_label': {},
        'overall': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0}
    }
    
    # Calculate metrics for each label
    for label in label_dict.keys():
        if label == 'O':  # Skip 'O' tag for individual metrics
            continue
            
        true_positives = sum(1 for p, t in zip(pred_labels, true_labels_str) 
                           if p == label and t == label)
        false_positives = sum(1 for p, t in zip(pred_labels, true_labels_str) 
                            if p == label and t != label)
        false_negatives = sum(1 for p, t in zip(pred_labels, true_labels_str) 
                            if p != label and t == label)
        
        # Update overall counts
        metrics['overall']['true_positives'] += true_positives
        metrics['overall']['false_positives'] += false_positives
        metrics['overall']['false_negatives'] += false_negatives
        
        # Calculate precision, recall, F1 for this label
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['per_label'][label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': true_positives + false_negatives
        }
    
    # Calculate overall metrics
    overall = metrics['overall']
    overall_precision = overall['true_positives'] / (overall['true_positives'] + overall['false_positives']) if (overall['true_positives'] + overall['false_positives']) > 0 else 0.0
    overall_recall = overall['true_positives'] / (overall['true_positives'] + overall['false_negatives']) if (overall['true_positives'] + overall['false_negatives']) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    metrics['overall'] = {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'true_positives': overall['true_positives'],
        'false_positives': overall['false_positives'],
        'false_negatives': overall['false_negatives']
    }
    
    return metrics

def select_features(features, labels, threshold=0.01):
    """Select most informative features using mutual information"""
    scores = mutual_info_classif(features, labels)
    return scores > threshold 