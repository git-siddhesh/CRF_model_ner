from crf_model import CRF
from data_loader import prepare_data, evaluate_predictions
import numpy as np
import json
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Prepare data
data = prepare_data(train_path='Train_CRF.csv', 
                   test_path='Test_CRF.csv',
                   val_path='ner_test.csv')

# Extract features and labels
train_features = data['train']['features']
train_labels = data['train']['labels']
test_features = data['test']['features']
test_labels = data['test']['labels']
val_features = data['validation']['features'] if data['validation'] else None
val_labels = data['validation']['labels'] if data['validation'] else None
label_dict = data['label_dict']
num_states = data['num_states']
num_features = data['num_features']

# Initialize model
l2_reg = 0.001  # Reduced L2 regularization

print(f"Number of states: {num_states}")
print(f"Number of features: {num_features}")
print(f"Training sequences: {len(train_features)}")
print(f"Validation sequences: {len(val_features) if val_features else 0}")
print(f"Test sequences: {len(test_features)}")

# Create and train model
model = CRF(num_states=num_states, num_features=num_features, l2_reg=l2_reg)

# Train with improved parameters
model.train(
    features_list=train_features,
    labels_list=train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    learning_rate=0.0005,  # Reduced learning rate
    num_iterations=300,    # Increased iterations
    patience=50,          # Increased patience
    min_iterations=150,   # Increased minimum iterations
    batch_size=16        # Small batch size for better generalization
)

# Make predictions on test set
predictions = []
for features in test_features:
    pred = model.predict(features)
    predictions.append(pred)

# Evaluate predictions
metrics = evaluate_predictions(test_labels, predictions, label_dict)

# Print overall metrics
print("\nOverall Metrics:")
print(f"Precision: {metrics['overall']['precision']:.4f}")
print(f"Recall: {metrics['overall']['recall']:.4f}")
print(f"F1 Score: {metrics['overall']['f1']:.4f}")

# Print per-label metrics
print("\nPer-label Metrics:")
for label, scores in metrics['per_label'].items():
    print(f"\n{label}:")
    print(f"Precision: {scores['precision']:.4f}")
    print(f"Recall: {scores['recall']:.4f}")
    print(f"F1 Score: {scores['f1']:.4f}")

# Save results
with open('results.json', 'w') as f:
    json.dump(metrics, f, indent=2) 