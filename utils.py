import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

def extract_features(sequence: List[str], pos_tags: List[str], feature_functions: List[callable]) -> np.ndarray:
    """
    Extract features from a sequence using provided feature functions
    
    Args:
        sequence: List of tokens/words
        pos_tags: List of POS tags
        feature_functions: List of feature functions that take sequence, pos_tags, and position as input
        
    Returns:
        Feature matrix of shape (sequence_length, num_features)
    """
    features = []
    for i in range(len(sequence)):
        position_features = []
        for feature_func in feature_functions:
            position_features.extend(feature_func(sequence, pos_tags, i))
        features.append(position_features)
    return np.array(features)

def create_word_features(sequence: List[str], pos_tags: List[str], position: int) -> List[float]:
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

def create_character_features(sequence: List[str], pos_tags: List[str], position: int) -> List[float]:
    """
    Create enhanced character-level features
    """
    features = []
    word = sequence[position]
    
    # Character type features
    features.extend([
        sum(c.isupper() for c in word) / max(len(word), 1),  # proportion of uppercase
        sum(c.isdigit() for c in word) / max(len(word), 1),  # proportion of digits
        sum(c.ispunctuation() if hasattr(c, 'ispunctuation') else c in '.,;:!?' for c in word) / max(len(word), 1),  # proportion of punctuation
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

def create_pos_features(sequence: List[str], pos_tags: List[str], position: int) -> List[float]:
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
        'proper': pos.startswith('NNP')
    }
    
    features.extend([1.0 if v else 0.0 for v in pos_categories.values()])
    
    # Previous POS features (window size 2)
    for i in range(2):
        prev_pos = position - (i + 1)
        if prev_pos >= 0:
            prev_tag = pos_tags[prev_pos]
            features.extend([
                1.0 if prev_tag.startswith(('NN', 'NNP', 'NNPS', 'NNS')) else 0.0,
                1.0 if prev_tag.startswith('NNP') else 0.0,
                1.0 if prev_tag.startswith(('JJ', 'JJR', 'JJS')) else 0.0,
                1.0 if prev_tag.startswith('IN') else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Next POS features (window size 2)
    for i in range(2):
        next_pos = position + (i + 1)
        if next_pos < len(pos_tags):
            next_tag = pos_tags[next_pos]
            features.extend([
                1.0 if next_tag.startswith(('NN', 'NNP', 'NNPS', 'NNS')) else 0.0,
                1.0 if next_tag.startswith('NNP') else 0.0,
                1.0 if next_tag.startswith(('JJ', 'JJR', 'JJS')) else 0.0,
                1.0 if next_tag.startswith('IN') else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
    
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
    
    Args:
        labels_list: List of label sequences
        label_dict: Dictionary mapping labels to indices
        
    Returns:
        List of label index arrays
    """
    return [np.array([label_dict[label] for label in labels]) for labels in labels_list] 