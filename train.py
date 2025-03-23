from crf_model import CRF
from data_loader import prepare_data, evaluate_predictions
import numpy as np
import json
import random
import pickle
import os
import matplotlib.pyplot as plt
import torch

# Set environment variables for MPS memory management
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.7'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.3'

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Function to clear GPU cache more aggressively
def clear_gpu_cache():
    if device.type == 'mps':
        import gc
        gc.collect()
        torch.mps.empty_cache()
        # Force garbage collection
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass
        gc.collect()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Set device with memory management
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Function to move tensors to device safely with mixed precision
def to_device(tensor, device):
    try:
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)
        return tensor.to(device)
    except RuntimeError as e:
        print(f"Memory error: {e}")
        clear_gpu_cache()
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)
        return tensor.to('cpu')  # Fallback to CPU if GPU memory is full

# Prepare data with memory management
try:
    data = prepare_data(
        train_path='data/ner_train.csv', 
        test_path='data/ner_test.csv',
        val_path=None,
        val_split=0.1,
        device=device,
        batch_size=16  # Reduced batch size
    )
except RuntimeError as e:
    print("Memory error encountered. Trying with smaller batch size...")
    clear_gpu_cache()
    try:
        data = prepare_data(
            train_path='data/ner_train.csv', 
            test_path='data/ner_test.csv',
            val_path=None,
            val_split=0.1,
            device=device,
            batch_size=8  # Further reduced batch size
        )
    except RuntimeError:
        print("Still encountering memory issues. Falling back to CPU...")
        device = torch.device("cpu")
        data = prepare_data(
            train_path='data/ner_train.csv', 
            test_path='data/ner_test.csv',
            val_path=None,
            val_split=0.1,
            device=device,
            batch_size=8
        )

# Convert numpy arrays to PyTorch tensors with pinned memory for faster transfer
def convert_to_tensors(features_list, labels_list):
    if device.type in ['cuda', 'mps']:
        return [torch.FloatTensor(f).pin_memory().to(dtype=torch.float16) for f in features_list], \
               [torch.LongTensor(l).pin_memory() for l in labels_list]
    return [torch.FloatTensor(f).to(dtype=torch.float16) for f in features_list], \
           [torch.LongTensor(l) for l in labels_list]

# Extract features and labels and convert to tensors
train_features, train_labels = convert_to_tensors(data['train']['features'], data['train']['labels'])
test_features, test_labels = convert_to_tensors(data['test']['features'], data['test']['labels'])
if data['validation']:
    val_features, val_labels = convert_to_tensors(data['validation']['features'], data['validation']['labels'])
else:
    val_features, val_labels = None, None

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

# Increase batch size for better GPU utilization with float16
batch_size = 8  # Reduced from 32
gradient_accumulation_steps = 4  # Accumulate gradients for 4 steps

# Create and train model with gradient accumulation
model = CRF(num_states=num_states, num_features=num_features, l2_reg=l2_reg, device=device)

# Enable automatic mixed precision training
if device.type == 'cuda':
    scaler = torch.cuda.amp.GradScaler()
else:
    # For MPS, we'll manually handle float16
    scaler = None

# Train with improved parameters and mixed precision
history = model.train(
    features_list=train_features,
    labels_list=train_labels,
    validation_features=val_features,
    validation_labels=val_labels,
    learning_rate=0.001,
    num_iterations=100,
    patience=20,
    min_iterations=30,
    batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps
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

# Save the trained model
model_path = 'models/crf_model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'label_dict': label_dict,
    'num_states': num_states,
    'num_features': num_features,
    'l2_reg': l2_reg
}, model_path)

print(f"\nModel saved to {model_path}")

# Make predictions on test set in batches
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    predictions = []
    for i in range(0, len(test_features), batch_size):
        batch_end = min(i + batch_size, len(test_features))
        # Stack features and move to device in one operation
        features_batch = torch.stack(test_features[i:batch_end]).to(device)
        pred = model.predict(features_batch)
        predictions.extend(pred.cpu().numpy())  # Move predictions back to CPU

# Free up memory
if device.type == 'mps':
    # For Apple Silicon
    import gc
    gc.collect()
    torch.mps.empty_cache()
elif device.type == 'cuda':
    # For NVIDIA GPU
    torch.cuda.empty_cache()

# Print memory usage if available
if device.type == 'cuda':
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
elif device.type == 'mps':
    print("\nUsing Apple Silicon GPU (MPS)")

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