# CRF Model Implementation

This is a pure Python implementation of a Conditional Random Field (CRF) model from scratch. The implementation includes:

- Core CRF model with forward-backward algorithm
- Viterbi algorithm for inference
- Feature extraction utilities
- Training using gradient descent

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use the CRF model:

```python
from crf_model import CRF
from utils import extract_features, create_word_features, create_character_features, create_label_dictionary, convert_labels_to_indices

# Example data
sequences = [
    ["John", "lives", "in", "New", "York"],
    ["Mary", "works", "at", "Google"]
]

labels = [
    ["PER", "O", "O", "LOC", "LOC"],
    ["PER", "O", "O", "ORG"]
]

# Create feature functions
feature_functions = [create_word_features, create_character_features]

# Extract features for all sequences
features_list = [extract_features(seq, feature_functions) for seq in sequences]

# Create label dictionary and convert labels to indices
label_dict = create_label_dictionary(labels)
labels_list = convert_labels_to_indices(labels, label_dict)

# Initialize and train the model
num_states = len(label_dict)
num_features = features_list[0].shape[1]
model = CRF(num_states, num_features)
model.train(features_list, labels_list, learning_rate=0.1, num_iterations=100)

# Make predictions
test_sequence = ["John", "works", "at", "Microsoft"]
test_features = extract_features(test_sequence, feature_functions)
predictions = model.predict(test_features)

# Convert predictions back to labels
predicted_labels = [list(label_dict.keys())[list(label_dict.values()).index(pred)] for pred in predictions]
print(predicted_labels)
```

## Features

The implementation includes:

1. **Core CRF Model**:
   - Forward-backward algorithm for computing partition function and marginals
   - Viterbi algorithm for inference
   - Gradient descent training

2. **Feature Extraction**:
   - Word-level features (capitalization, numbers, length)
   - Character-level features (n-gram prefixes and suffixes)
   - Context features (previous and next word information)

3. **Utilities**:
   - Label dictionary creation
   - Feature extraction pipeline
   - Data preprocessing functions

## Customization

You can customize the model by:

1. Adding new feature functions in `utils.py`
2. Modifying the learning rate and number of iterations in the training process
3. Adjusting the feature extraction pipeline to include domain-specific features

## Dependencies

- numpy >= 1.21.0 