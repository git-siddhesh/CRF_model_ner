import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import ast
from sklearn.feature_selection import mutual_info_classif
import torch

# Global variables for feature extraction
WORD_VOCAB = {}  # Will be initialized during data loading
POS_VOCAB = {}   # Will be initialized during data loading
UNK_IDX = 0      # Index for unknown tokens

def initialize_vocabularies(sequences: List[List[str]], pos_tags: List[List[str]], min_word_freq: int = 3):
    """Initialize word and POS tag vocabularies with frequency thresholding"""
    global WORD_VOCAB, POS_VOCAB
    
    # Initialize word vocabulary with frequency counting
    word_freq = {}
    for sequence in sequences:
        for word in sequence:
            word = word.lower()
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter words by frequency threshold
    frequent_words = {word for word, freq in word_freq.items() if freq >= min_word_freq}
    WORD_VOCAB = {word: idx + 1 for idx, word in enumerate(sorted(frequent_words))}  # 0 reserved for UNK
    
    # Initialize POS vocabulary (keep all POS tags as they are usually limited)
    pos_set = set()
    for pos_sequence in pos_tags:
        for pos in pos_sequence:
            pos_set.add(pos)
    POS_VOCAB = {pos: idx for idx, pos in enumerate(sorted(pos_set))}
    
    # Calculate total number of features
    global NUM_FEATURES
    NUM_FEATURES = (len(WORD_VOCAB) + 1 +  # Add 1 for UNK token
                   len(POS_VOCAB) +
                   4)  # Additional features: capitalization (3) + numeric (1)
    
    print(f"Vocabulary sizes after frequency filtering (min_freq={min_word_freq}):")
    print(f"Words: {len(WORD_VOCAB)} (reduced from {len(word_freq)})")
    print(f"POS tags: {len(POS_VOCAB)}")
    print(f"Total features: {NUM_FEATURES}")

def load_csv_data(file_path: str) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    """
    Load data from CSV file and convert to sequences of words, POS tags, and labels.
    Handles mismatched sequences and sorts by length for efficient batching.
    """
    df = pd.read_csv(file_path)
    
    sequences = []
    pos_tags = []
    labels = []
    
    skipped = 0
    fixed = 0
    
    for _, row in df.iterrows():
        # Convert string representations of lists to actual lists
        sentence = row['Sentence'].split()
        pos = ast.literal_eval(row['POS'])
        tags = ast.literal_eval(row['Tag'])
        
        # Handle mismatched lengths
        min_len = min(len(sentence), len(pos), len(tags))
        max_len = max(len(sentence), len(pos), len(tags))
        
        if min_len != max_len:
            if abs(len(sentence) - len(pos)) <= 2 and abs(len(sentence) - len(tags)) <= 2:
                # Fix minor misalignments by truncating to shortest length
                sentence = sentence[:min_len]
                pos = pos[:min_len]
                tags = tags[:min_len]
                fixed += 1
            else:
                # Skip sequences with major misalignments
                skipped += 1
                continue
            
        sequences.append(sentence)
        pos_tags.append(pos)
        labels.append(tags)
    
    if skipped > 0 or fixed > 0:
        print(f"\nSequence length handling:")
        print(f"- Fixed {fixed} sequences with minor misalignments")
        print(f"- Skipped {skipped} sequences with major misalignments")
        print(f"- Retained {len(sequences)} valid sequences")
    
    # Sort sequences by length for more efficient batching
    sorted_indices = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    sequences = [sequences[i] for i in sorted_indices]
    pos_tags = [pos_tags[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    
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

def extract_features(sequence: List[str], pos_tags: List[str]) -> torch.Tensor:
    """
    Extract features for a sequence using PyTorch tensors
    """
    features = []
    for i, (word, pos) in enumerate(zip(sequence, pos_tags)):
        # Word features
        word_lower = word.lower()
        
        # Initialize feature vector with zeros
        feature_vec = torch.zeros(NUM_FEATURES)
        
        # Word identity features
        feature_vec[WORD_VOCAB.get(word_lower, UNK_IDX)] = 1.0
        
        # POS tag features
        feature_vec[len(WORD_VOCAB) + POS_VOCAB.get(pos, UNK_IDX)] = 1.0
        
        # Capitalization features
        if word[0].isupper():
            feature_vec[-4] = 1.0
        if word.isupper():
            feature_vec[-3] = 1.0
        if any(c.isupper() for c in word[1:]):
            feature_vec[-2] = 1.0
            
        # Numeric feature
        if any(c.isdigit() for c in word):
            feature_vec[-1] = 1.0
            
        features.append(feature_vec)
    
    return torch.stack(features)

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

def prepare_data(train_path: str, test_path: str, val_path: str = None, 
              val_split: float = 0.1, device: str = None, batch_size: int = 32) -> Dict:
    """
    Prepare data for CRF model training and evaluation with GPU support and memory management
    """
    # Set device
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load all data first
    train_sequences, train_pos, train_labels = load_csv_data(train_path)
    test_sequences, test_pos, test_labels = load_csv_data(test_path)
    
    # Initialize vocabularies
    all_sequences = train_sequences + test_sequences
    all_pos_tags = train_pos + test_pos
    initialize_vocabularies(all_sequences, all_pos_tags)
    
    print(f"Vocabulary sizes - Words: {len(WORD_VOCAB)}, POS tags: {len(POS_VOCAB)}")
    print(f"Total number of features: {NUM_FEATURES}")
    
    # Process data in batches
    def process_batch(sequences, pos_tags, start_idx, end_idx, device):
        """Process a batch of sequences with better memory management using float16"""
        batch_features = []
        for seq, pos in zip(sequences[start_idx:end_idx], pos_tags[start_idx:end_idx]):
            try:
                # Process each sequence
                features = extract_features(seq, pos)
                # Convert to float16 tensor efficiently
                features_tensor = features.clone().detach().to(dtype=torch.float16)
                
                if device.type == 'mps':
                    try:
                        features_tensor = features_tensor.to(device)
                    except RuntimeError:
                        print(".", end="", flush=True)
                        if torch.mps.is_available():
                            torch.mps.empty_cache()
                        features_tensor = features_tensor.cpu()
                else:
                    features_tensor = features_tensor.to(device)
                
                batch_features.append(features_tensor)
                
                # Clear cache more frequently
                if len(batch_features) % 5 == 0:  # Reduced from 10 to 5
                    if device.type == 'mps':
                        import gc
                        gc.collect()
                        torch.mps.empty_cache()
                    elif device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"\nWarning: Error processing sequence: {e}")
                # Fallback to CPU processing with float16
                features = extract_features(seq, pos)
                features_tensor = features.clone().detach().to(dtype=torch.float16).cpu()
                batch_features.append(features_tensor)
            
        return batch_features
    
    # Process training data in batches
    train_features = []
    for i in range(0, len(train_sequences), batch_size):
        end_idx = min(i + batch_size, len(train_sequences))
        batch_features = process_batch(train_sequences, train_pos, i, end_idx, device)
        train_features.extend(batch_features)
        
        # Clear cache after each batch
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Process test data in batches
    test_features = []
    for i in range(0, len(test_sequences), batch_size):
        end_idx = min(i + batch_size, len(test_sequences))
        batch_features = process_batch(test_sequences, test_pos, i, end_idx, device)
        test_features.extend(batch_features)
        
        # Clear cache after each batch
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Create label dictionary
    label_dict = create_label_dictionary(train_labels + test_labels)
    
    # Convert labels to indices
    train_labels_indices = [torch.tensor(convert_labels_to_indices([labels], label_dict)[0], 
                                       dtype=torch.long, device=device)
                          for labels in train_labels]
    test_labels_indices = [torch.tensor(convert_labels_to_indices([labels], label_dict)[0],
                                      dtype=torch.long, device=device)
                         for labels in test_labels]
    
    # Handle validation data
    val_data = None
    if val_path:
        val_sequences, val_pos, val_labels = load_csv_data(val_path)
        # Process validation data in batches
        val_features = []
        for i in range(0, len(val_sequences), batch_size):
            end_idx = min(i + batch_size, len(val_sequences))
            batch_features = process_batch(val_sequences, val_pos, i, end_idx, device)
            val_features.extend(batch_features)
            
            # Clear cache after each batch
            if device.type == 'mps':
                torch.mps.empty_cache()
                
        val_labels_indices = [torch.tensor(convert_labels_to_indices([labels], label_dict)[0],
                                         dtype=torch.long, device=device)
                            for labels in val_labels]
        val_data = {
            'features': val_features,
            'labels': val_labels_indices,
            'sequences': val_sequences,
            'pos_tags': val_pos,
            'original_labels': val_labels
        }
    else:
        # Split training data
        num_val = int(len(train_sequences) * val_split)
        indices = torch.randperm(len(train_sequences))
        val_indices = indices[:num_val].tolist()
        train_indices = indices[num_val:].tolist()
        
        # Update training and create validation sets
        val_features = [train_features[i] for i in val_indices]
        val_labels_indices = [train_labels_indices[i] for i in val_indices]
        val_sequences = [train_sequences[i] for i in val_indices]
        val_pos = [train_pos[i] for i in val_indices]
        val_labels = [train_labels[i] for i in val_indices]
        
        # Update training set
        train_features = [train_features[i] for i in train_indices]
        train_labels_indices = [train_labels_indices[i] for i in train_indices]
        train_sequences = [train_sequences[i] for i in train_indices]
        train_pos = [train_pos[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
        
        val_data = {
            'features': val_features,
            'labels': val_labels_indices,
            'sequences': val_sequences,
            'pos_tags': val_pos,
            'original_labels': val_labels
        }
        
        print(f"Split training data: {len(train_sequences)} train, {len(val_sequences)} validation sequences")
    
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
        'num_features': train_features[0].shape[1] if train_features else NUM_FEATURES,
        'num_states': len(label_dict),
        'device': device
    }

def evaluate_predictions(true_labels: List[torch.Tensor], pred_labels: List[torch.Tensor], 
                       label_dict: Dict[str, int]) -> Dict:
    """
    Evaluate predictions using PyTorch tensors
    """
    # Convert tensors to CPU for evaluation
    true_labels = [labels.cpu() for labels in true_labels]
    pred_labels = [labels.cpu() for labels in pred_labels]
    
    # Create reverse label dictionary
    rev_label_dict = {v: k for k, v in label_dict.items()}
    
    metrics = {
        'overall': {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0},
        'per_label': {label: {'true_positives': 0, 'false_positives': 0, 
                             'false_negatives': 0, 'support': 0}
                     for label in label_dict}
    }
    
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true_label, pred_label in zip(true_seq, pred_seq):
            true_label_str = rev_label_dict[true_label.item()]
            pred_label_str = rev_label_dict[pred_label.item()]
            
            # Update per-label metrics
            metrics['per_label'][true_label_str]['support'] += 1
            if true_label_str == pred_label_str:
                if true_label_str != 'O':  # Only count non-O labels
                    metrics['overall']['true_positives'] += 1
                    metrics['per_label'][true_label_str]['true_positives'] += 1
            else:
                if true_label_str != 'O':
                    metrics['overall']['false_negatives'] += 1
                    metrics['per_label'][true_label_str]['false_negatives'] += 1
                if pred_label_str != 'O':
                    metrics['overall']['false_positives'] += 1
                    metrics['per_label'][pred_label_str]['false_positives'] += 1
    
    # Calculate precision, recall, and F1 score
    def calculate_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    
    # Overall metrics
    precision, recall, f1 = calculate_metrics(
        metrics['overall']['true_positives'],
        metrics['overall']['false_positives'],
        metrics['overall']['false_negatives']
    )
    metrics['overall'].update({'precision': precision, 'recall': recall, 'f1': f1})
    
    # Per-label metrics
    for label in metrics['per_label']:
        precision, recall, f1 = calculate_metrics(
            metrics['per_label'][label]['true_positives'],
            metrics['per_label'][label]['false_positives'],
            metrics['per_label'][label]['false_negatives']
        )
        metrics['per_label'][label].update({'precision': precision, 'recall': recall, 'f1': f1})
    
    return metrics

def select_features(features, labels, threshold=0.01):
    """Select most informative features using mutual information"""
    scores = mutual_info_classif(features, labels)
    return scores > threshold 