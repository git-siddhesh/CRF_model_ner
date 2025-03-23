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


---

I'll explain the CRF training pipeline from a theoretical perspective:

1. **Conditional Random Fields (CRF) Theory**:
- CRF is a discriminative probabilistic model that models P(Y|X) directly
- For sequence labeling: X = input sequence, Y = label sequence
- The probability is modeled as:

\[
P(Y|X) = \frac{1}{Z(X)} \exp(\sum_{t=1}^T \sum_k \lambda_k f_k(y_t, y_{t-1}, X, t))
\]

where:
- Z(X) is the partition function (normalization constant)
- f_k are feature functions
- λ_k are weights to be learned
- T is sequence length

2. **Feature Engineering**:
```python
def _initialize_feature_templates(self):
    templates = [
        # Current word features
        {'type': 'word', 'offset': 0},
        {'type': 'prefix', 'offset': 0, 'length': 3},
        {'type': 'suffix', 'offset': 0, 'length': 3},
        # ... more features
    ]
```
The model uses several types of features:
- **Local Features**: 
  - Word-level (case, prefixes, suffixes)
  - Character-level patterns
  - POS tags
- **Contextual Features**:
  - Window-based features (previous/next words)
  - N-gram patterns
- **Structural Features**:
  - Transition patterns
  - BIO scheme constraints

3. **Model Architecture**:

a) **Emission Scores (State Features)**:
\[
\phi_{\text{emit}}(x_t, y_t) = w_{\text{emit}}^T f(x_t, y_t) + b_{\text{emit}}
\]

b) **Transition Scores**:
\[
\phi_{\text{trans}}(y_{t-1}, y_t) = w_{\text{trans}}^T g(y_{t-1}, y_t) + b_{\text{trans}}
\]

c) **Second-order Transitions**:
\[
\phi_{\text{2nd}}(y_{t-2}, y_{t-1}, y_t) = w_{\text{2nd}}^T h(y_{t-2}, y_{t-1}, y_t) + b_{\text{2nd}}
\]

4. **Training Objective**:

The negative log-likelihood loss with regularization:
\[
L(\theta) = -\sum_{i=1}^N \log P(Y^{(i)}|X^{(i)}; \theta) + \frac{\lambda}{2}||\theta||^2
\]

Components:
```python
def _compute_crf_loss(self, state_features, transition_features, labels, log_partition):
```
- **Likelihood Term**: Probability of correct sequence
- **Partition Function**: Normalization over all possible sequences
- **L2 Regularization**: Prevents overfitting
- **Label Smoothing**: Prevents overconfident predictions
- **BIO Constraints**: Domain-specific structural constraints

5. **Forward-Backward Algorithm**:

a) **Forward Pass**:
\[
\alpha_t(j) = \sum_i \alpha_{t-1}(i) \exp(\phi_{\text{trans}}(i,j) + \phi_{\text{emit}}(x_t,j))
\]

b) **Backward Pass**:
\[
\beta_t(i) = \sum_j \beta_{t+1}(j) \exp(\phi_{\text{trans}}(i,j) + \phi_{\text{emit}}(x_{t+1},j))
\]

6. **Optimization Strategy**:

```python
def train(self, features_list, labels_list, ...):
```

a) **Learning Rate Schedule**:
- Warmup phase: Linear increase
- Cosine decay: Smooth learning rate reduction
\[
\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})(1 + \cos(\pi \cdot \text{progress}))
\]

b) **Adam Optimizer**:
- Adaptive moment estimation
- First moment (mean): \[ m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \]
- Second moment (variance): \[ v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \]
- Update rule: \[ \theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \]

7. **Numerical Stability**:

a) **Log-Space Computations**:
- Forward-backward in log space
- Log-sum-exp trick:
\[
\log\sum_i \exp(x_i) = \max_i(x_i) + \log\sum_i \exp(x_i - \max_i(x_i))
\]

b) **Feature Normalization**:
```python
def _normalize_features(self, features):
```
- Z-score normalization
- Gradient clipping
- Value bounds enforcement

8. **Training Pipeline Flow**:

1. **Data Preparation**:
   - Feature extraction
   - Label encoding
   - Batch creation

2. **Forward Pass**:
   - Compute state features
   - Compute transition scores
   - Forward-backward algorithm
   - Calculate loss

3. **Backward Pass**:
   - Compute gradients
   - Apply BIO constraints
   - Update parameters via Adam

4. **Regularization**:
   - L2 weight decay
   - Label smoothing
   - Feature normalization
   - Gradient clipping

5. **Monitoring**:
   - Loss tracking
   - Validation metrics
   - Early stopping

The training process combines:
- Structured prediction (CRF)
- Deep learning optimization techniques
- Domain-specific constraints (BIO scheme)
- Numerical stability considerations

This creates a robust model for sequence labeling tasks like Named Entity Recognition (NER).

Would you like me to elaborate on any specific theoretical aspect?
