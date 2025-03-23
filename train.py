from crf_model import CRF
from data_loader import prepare_data, evaluate_predictions
import numpy as np
import json
import random
import pickle
import os
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Prepare data
data = prepare_data(train_path='data/ner_train.csv', 
                   test_path='data/ner_test.csv',
                   val_path=None,  # Use split from training data
                   val_split=0.1)  # Use 10% of training data for validation

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
history = model.train(
    features_list=train_features,
    labels_list=train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    learning_rate=0.001,    # Initial learning rate
    num_iterations=100,     # Moderate number of iterations
    patience=20,           # Shorter patience for quicker stopping
    min_iterations=30,     # Lower minimum iterations
    batch_size=16         # Smaller batch size for better gradient estimates
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('CRF Model Training History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot
plot_path = 'plots/training_history.png'
plt.savefig(plot_path)
plt.close()

print(f"\nTraining history plot saved to {plot_path}")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model
model_path = 'models/crf_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save the label dictionary separately for easier access
label_dict_path = 'models/label_dict.pkl'
with open(label_dict_path, 'wb') as f:
    pickle.dump(label_dict, f)

print(f"\nModel saved to {model_path}")
print(f"Label dictionary saved to {label_dict_path}")

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
print(f"True Positives: {metrics['overall']['true_positives']}")
print(f"False Positives: {metrics['overall']['false_positives']}")
print(f"False Negatives: {metrics['overall']['false_negatives']}")

# Print per-label metrics
print("\nPer-label Metrics:")
for label, scores in metrics['per_label'].items():
    print(f"\n{label}:")
    print(f"Precision: {scores['precision']:.4f}")
    print(f"Recall: {scores['recall']:.4f}")
    print(f"F1 Score: {scores['f1']:.4f}")
    print(f"Support: {scores['support']}")

# Save results
with open('results.json', 'w') as f:
    json.dump(metrics, f, indent=2) 