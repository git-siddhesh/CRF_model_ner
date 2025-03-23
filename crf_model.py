import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import math
import torch.optim as optim

class CRF(nn.Module):
    def __init__(self, num_states: int, num_features: int, l2_reg: float = 0.1, device: torch.device = None):
        """
        Initialize CRF model with second-order transitions
        """
        super(CRF, self).__init__()
        self.num_states = num_states
        self.num_features = num_features
        self.l2_reg = l2_reg
        
        # Set device (GPU if available, else CPU)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize label dictionary
        self.label_dict = None  # Will be set during training
        
        # Initialize weights with improved Xavier/Glorot initialization
        scale = np.sqrt(6.0 / (num_features + num_states))
        
        # Convert all parameters to PyTorch tensors and move to device
        self.state_weights = nn.Parameter(torch.FloatTensor(num_states, num_features).uniform_(-scale, scale))
        self.transition_weights = nn.Parameter(torch.FloatTensor(num_states, num_states).uniform_(-scale, scale))
        self.second_order_weights = nn.Parameter(torch.FloatTensor(num_states, num_states, num_states).uniform_(-scale, scale))
        
        self.state_bias = nn.Parameter(torch.zeros(num_states))
        self.transition_bias = nn.Parameter(torch.zeros(num_states, num_states))
        self.second_order_bias = nn.Parameter(torch.zeros(num_states, num_states, num_states))
        
        # Move model to device
        self.to(self.device)
        
        # Initialize feature templates
        self.feature_templates = self._initialize_feature_templates()
        
        # Initialize feature statistics for normalization
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
        # Initialize best model weights for early stopping
        self.register_buffer('best_state_weights', None)
        self.register_buffer('best_transition_weights', None)
        self.register_buffer('best_state_bias', None)
        self.register_buffer('best_transition_bias', None)
        
        # Initialize Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Initialize Adam moments and velocities for weights and biases
        self.state_m = torch.zeros_like(self.state_weights)
        self.state_v = torch.zeros_like(self.state_weights)
        self.state_bias_m = torch.zeros_like(self.state_bias)
        self.state_bias_v = torch.zeros_like(self.state_bias)
        self.transition_m = torch.zeros_like(self.transition_weights)
        self.transition_v = torch.zeros_like(self.transition_weights)
        self.transition_bias_m = torch.zeros_like(self.transition_bias)
        self.transition_bias_v = torch.zeros_like(self.transition_bias)
        
        # Initialize iteration counter for Adam
        self.t = 0
        
        # Initialize parameters with float16
        self.transition = nn.Parameter(torch.zeros(num_states, num_states, dtype=torch.float16))
        self.feature_weights = nn.Parameter(torch.zeros(num_features, num_states, dtype=torch.float16))
        
        # Initialize optimizer with higher epsilon for float16 stability
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, eps=1e-4)
        
    def _initialize_feature_templates(self) -> List[Dict]:
        """
        Initialize feature templates for enhanced feature extraction
        """
        templates = [
            # Current word features
            {'type': 'word', 'offset': 0},
            {'type': 'prefix', 'offset': 0, 'length': 3},
            {'type': 'suffix', 'offset': 0, 'length': 3},
            {'type': 'is_capitalized', 'offset': 0},
            {'type': 'has_number', 'offset': 0},
            {'type': 'has_hyphen', 'offset': 0},
            
            # Previous word features
            {'type': 'word', 'offset': -1},
            {'type': 'is_capitalized', 'offset': -1},
            
            # Next word features
            {'type': 'word', 'offset': 1},
            {'type': 'is_capitalized', 'offset': 1},
            
            # POS features
            {'type': 'pos', 'offset': 0},
            {'type': 'pos', 'offset': -1},
            {'type': 'pos', 'offset': 1},
            
            # POS bigrams
            {'type': 'pos_bigram', 'offset': -1},
            {'type': 'pos_bigram', 'offset': 0},
            
            # Word shape features
            {'type': 'word_shape', 'offset': 0},
            {'type': 'word_pattern', 'offset': 0},
        ]
        return templates
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features to have zero mean and unit variance
        """
        if self.feature_mean is None:
            self.feature_mean = features.mean(0)
            self.feature_std = features.std(0)
            self.feature_std[self.feature_std == 0] = 1  # Avoid division by zero
            
        normalized = (features - self.feature_mean) / self.feature_std
        return torch.clamp(normalized, -10, 10)
    
    def _clip_gradients(self, gradients: torch.Tensor, max_norm: float = 1.0) -> torch.Tensor:
        """
        Clip gradients to prevent exploding gradients
        """
        norm = torch.linalg.norm(gradients)
        if norm > max_norm:
            gradients = gradients * (max_norm / norm)
        return gradients
    
    def _compute_state_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute state feature scores with bias terms
        """
        normalized_features = self._normalize_features(features)
        scores = torch.matmul(normalized_features, self.state_weights.t()) + self.state_bias
        return torch.clamp(scores, -5, 5)
    
    def _compute_transition_features(self, prev_state=None, prev_prev_state=None) -> torch.Tensor:
        """
        Compute transition feature scores with second-order transitions
        """
        if prev_prev_state is None or prev_state is None:
            return torch.clamp(self.transition_weights + self.transition_bias, -5, 5)
        
        first_order = self.transition_weights + self.transition_bias
        second_order = self.second_order_weights[prev_prev_state, prev_state] + self.second_order_bias[prev_prev_state, prev_state]
        
        return torch.clamp(first_order + second_order, -5, 5)
    
    def _log_sum_exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log(sum(exp(x))) in a numerically stable way
        """
        max_x = x.max()
        return max_x + torch.log(torch.sum(torch.exp(x - max_x)))
    
    def _forward_algorithm(self, state_features: torch.Tensor, transition_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward algorithm for computing partition function and forward messages in log space
        """
        sequence_length = state_features.shape[0]
        alpha = torch.zeros(sequence_length, self.num_states, device=self.device)
        
        # Initialize first position
        alpha[0] = state_features[0]
        
        # Forward pass in log space
        for t in range(1, sequence_length):
            for j in range(self.num_states):
                scores = alpha[t-1] + transition_features[:, j] + state_features[t, j]
                alpha[t, j] = self._log_sum_exp(scores)
        
        log_partition = self._log_sum_exp(alpha[-1])
        return log_partition, alpha
    
    def _backward_algorithm(self, state_features: torch.Tensor, transition_features: torch.Tensor) -> torch.Tensor:
        """
        Backward algorithm for computing backward messages in log space
        """
        sequence_length = state_features.shape[0]
        beta = torch.zeros(sequence_length, self.num_states, device=self.device)
        
        # Initialize last position
        beta[-1] = 0.0
        
        # Backward pass in log space
        for t in range(sequence_length-2, -1, -1):
            for i in range(self.num_states):
                scores = beta[t+1] + transition_features[i, :] + state_features[t+1]
                beta[t, i] = self._log_sum_exp(scores)
        
        return beta
    
    def _compute_marginals(self, alpha: torch.Tensor, beta: torch.Tensor, log_partition: torch.Tensor) -> torch.Tensor:
        """
        Compute marginal probabilities in log space with numerical stability
        """
        # Compute log marginals
        log_marginals = alpha + beta - log_partition
        
        # Convert to probabilities with numerical stability
        max_vals = log_marginals.max(1, keepdim=True)
        exp_vals = torch.exp(log_marginals - max_vals)
        marginals = exp_vals / exp_vals.sum(1, keepdim=True)
        
        return marginals
    
    def _compute_pairwise_marginals(self, alpha: torch.Tensor, beta: torch.Tensor, 
                                  state_features: torch.Tensor, transition_features: torch.Tensor,
                                  log_partition: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise marginal probabilities with numerical stability
        """
        sequence_length = alpha.shape[0]
        pairwise_marginals = torch.zeros(sequence_length-1, self.num_states, self.num_states, device=self.device)
        
        for t in range(sequence_length-1):
            # Compute log pairwise marginals
            log_pairwise = (alpha[t, :, None] + 
                           transition_features + 
                           state_features[t+1] + 
                           beta[t+1] - 
                           log_partition)
            
            # Convert to probabilities with numerical stability
            max_val = log_pairwise.max()
            exp_vals = torch.exp(log_pairwise - max_val)
            pairwise_marginals[t] = exp_vals / exp_vals.sum()
        
        return pairwise_marginals
    
    def _compute_crf_loss(self, state_features: torch.Tensor, transition_features: torch.Tensor,
                         labels: torch.Tensor, log_partition: torch.Tensor) -> torch.Tensor:
        """
        Compute CRF loss with proper scaling
        """
        sequence_length = len(labels)
        
        # Scale down features to prevent explosion
        state_features = torch.clamp(state_features, -10, 10)
        transition_features = torch.clamp(transition_features, -10, 10)
        
        # Compute unary potentials (state scores)
        unary_score = torch.sum(state_features[torch.arange(sequence_length), labels])
        
        # Compute pairwise potentials (transition scores)
        pairwise_score = 0
        if sequence_length > 1:
            for t in range(sequence_length - 1):
                current_label = labels[t]
                next_label = labels[t + 1]
                
                transition_score = transition_features[current_label, next_label]
                
                if self._violates_bio_constraints(current_label, next_label):
                    transition_score -= 10.0
                    
                pairwise_score += transition_score
        
        sequence_score = unary_score + pairwise_score
        
        # Compute loss with label smoothing
        epsilon = 0.1
        smooth_loss = (1 - epsilon) * (-sequence_score + log_partition) + \
                     epsilon * self._compute_label_smoothing_loss(state_features, labels)
        
        return smooth_loss / sequence_length

    def _violates_bio_constraints(self, current_label: int, next_label: int) -> bool:
        """
        Check if the transition violates BIO scheme constraints using integer labels
        """
        try:
            # Convert integer indices to BIO labels
            label_list = list(self.label_dict.keys())
            current_tag = label_list[current_label]
            next_tag = label_list[next_label]
            
            # O can be followed by O or B-*
            if current_tag == 'O':
                return next_tag.startswith('I-')
            
            # B-X can be followed by I-X or O or B-*
            if current_tag.startswith('B-'):
                entity_type = current_tag[2:]
                if next_tag.startswith('I-'):
                    return next_tag[2:] != entity_type
                return False
            
            # I-X can be followed by I-X or O or B-*
            if current_tag.startswith('I-'):
                entity_type = current_tag[2:]
                if next_tag.startswith('I-'):
                    return next_tag[2:] != entity_type
                return False
            
            return False
            
        except (AttributeError, IndexError, TypeError) as e:
            print(f"Warning: Error in BIO constraints check - {str(e)}")
            print(f"Current label: {current_label}, Next label: {next_label}")
            print(f"Label dictionary: {self.label_dict}")
            return False  # Default to allowing the transition if there's an error

    def _compute_label_smoothing_loss(self, state_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss to prevent overconfident predictions
        """
        uniform_distribution = torch.ones(self.num_states, device=self.device) / self.num_states
        kl_divergence = 0.0
        
        for t in range(len(labels)):
            logits = state_features[t]
            probs = torch.exp(logits - self._log_sum_exp(logits))
            kl_divergence += torch.sum(uniform_distribution * torch.log(uniform_distribution / (probs + 1e-10)))
        
        return kl_divergence / len(labels)

    def _update_weights_adam(self, gradients: torch.Tensor, weights: torch.Tensor, 
                              m: torch.Tensor, v: torch.Tensor, learning_rate: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update weights using Adam optimizer
        
        Args:
            gradients: Weight gradients
            weights: Current weights
            m: First moment vector
            v: Second moment vector
            learning_rate: Learning rate
            
        Returns:
            Tuple of (updated weights, updated first moment, updated second moment)
        """
        self.t += 1
        
        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * gradients
        
        # Update biased second raw moment estimate
        v = self.beta2 * v + (1 - self.beta2) * gradients ** 2
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.beta2**self.t)
        
        # Update weights
        weights = weights - learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        
        return weights, m, v

    def train(self, features_list: List[torch.Tensor], labels_list: List[torch.Tensor],
              learning_rate: float = 0.001, num_iterations: int = 100, patience: int = 10,
              min_iterations: int = 20, validation_features: List[torch.Tensor] = None,
              validation_labels: List[torch.Tensor] = None, batch_size: int = 32,
              gradient_accumulation_steps: int = 4):
        """
        Train the CRF model with gradient accumulation and improved memory management
        """
        self.train()  # Set model to training mode
        
        # Initialize optimizer with learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-4)
        
        # Initialize training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Adjust learning rate for gradient accumulation
        effective_batch_size = batch_size * gradient_accumulation_steps
        adjusted_lr = learning_rate * (effective_batch_size / 32)  # Scale learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=adjusted_lr, eps=1e-4)
        
        for iteration in range(num_iterations):
            total_loss = 0
            num_batches = 0
            self.optimizer.zero_grad()  # Zero gradients at start of iteration
            
            # Process in batches with gradient accumulation
            for i in range(0, len(features_list), batch_size):
                batch_end = min(i + batch_size, len(features_list))
                features_batch = features_list[i:batch_end]
                labels_batch = labels_list[i:batch_end]
                
                try:
                    # Convert to float16 and move to device
                    features_tensor = torch.stack([f.to(dtype=torch.float16) for f in features_batch]).to(self.device)
                    labels_tensor = torch.stack(labels_batch).to(self.device)
                    
                    # Forward pass
                    scores = self(features_tensor)
                    
                    # Compute loss
                    loss = -self._compute_log_likelihood(scores, labels_tensor)
                    if self.l2_reg > 0:
                        l2_loss = self.l2_reg * (torch.norm(self.feature_weights) + torch.norm(self.transition))
                        loss += l2_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update total loss
                    total_loss += loss.item() * gradient_accumulation_steps
                    num_batches += 1
                    
                    # Step optimizer after accumulating gradients
                    if (i + batch_size) % (batch_size * gradient_accumulation_steps) == 0 or (i + batch_size) >= len(features_list):
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    # Clear memory
                    del features_tensor, labels_tensor, scores, loss
                    if (i + batch_size) % (batch_size * 5) == 0:
                        if self.device.type == 'mps':
                            import gc
                            gc.collect()
                            torch.mps.empty_cache()
                    
                except RuntimeError as e:
                    print(f"Error in batch {i}: {e}")
                    if self.device.type == 'mps':
                        import gc
                        gc.collect()
                        torch.mps.empty_cache()
                    continue
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            history['train_loss'].append(avg_loss)
            
            # Validation with memory optimization
            if validation_features and validation_labels:
                val_loss = self._compute_validation_loss(validation_features, validation_labels, batch_size)
                history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience and iteration >= min_iterations:
                    print(f"Early stopping at iteration {iteration}")
                    break
            
            # Print progress and clear memory
            if (iteration + 1) % 10 == 0:
                val_msg = f", Validation Loss: {val_loss:.4f}" if validation_features else ""
                print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}{val_msg}")
                if self.device.type == 'mps':
                    import gc
                    gc.collect()
                    torch.mps.empty_cache()
        
        return history

    def _compute_validation_loss(self, validation_features: List[torch.Tensor],
                               validation_labels: List[torch.Tensor], batch_size: int) -> float:
        """
        Compute validation loss with memory optimization
        """
        self.eval()  # Set model to evaluation mode
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(validation_features), batch_size):
                batch_end = min(i + batch_size, len(validation_features))
                features_batch = validation_features[i:batch_end]
                labels_batch = validation_labels[i:batch_end]
                
                try:
                    # Convert to float16 and move to device
                    features_tensor = torch.stack([f.to(dtype=torch.float16) for f in features_batch]).to(self.device)
                    labels_tensor = torch.stack(labels_batch).to(self.device)
                    
                    # Compute loss
                    scores = self(features_tensor)
                    loss = -self._compute_log_likelihood(scores, labels_tensor)
                    
                    val_loss += loss.item()
                    num_batches += 1
                    
                    # Clear memory
                    del features_tensor, labels_tensor, scores, loss
                    if (i + batch_size) % (batch_size * 5) == 0:
                        if self.device.type == 'mps':
                            import gc
                            gc.collect()
                            torch.mps.empty_cache()
                
                except RuntimeError as e:
                    print(f"Error in validation batch {i}: {e}")
                    continue
        
        self.train()  # Set model back to training mode
        return val_loss / num_batches if num_batches > 0 else float('inf')
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict labels using Viterbi algorithm
        """
        with torch.no_grad():
            features = features.to(self.device)
            state_features = self._compute_state_features(features)
            sequence_length = features.shape[0]
            
            # Initialize Viterbi variables
            viterbi = torch.zeros(sequence_length, self.num_states, self.num_states, device=self.device)
            backpointer = torch.zeros(sequence_length, self.num_states, self.num_states, 
                                    dtype=torch.long, device=self.device)
            
            # Initialize first position
            viterbi[0] = state_features[0].unsqueeze(1) + self.transition_weights
            
            # Forward pass
            for t in range(2, sequence_length):
                for k in range(self.num_states):
                    for j in range(self.num_states):
                        scores = (viterbi[t-1, :, j] + 
                                self.transition_weights[:, k] +
                                self.second_order_weights[:, j, k] +
                                state_features[t, k])
                        viterbi[t, j, k], backpointer[t, j, k] = scores.max(0)
            
            # Backtrack
            labels = torch.zeros(sequence_length, dtype=torch.long, device=self.device)
            
            # Find best last two states
            last_scores = viterbi[-1]
            labels[-2], labels[-1] = torch.unravel_index(last_scores.argmax(), last_scores.shape)
            
            # Backtrack remaining states
            for t in range(sequence_length-3, -1, -1):
                labels[t] = backpointer[t+2, labels[t+1], labels[t+2]]
            
            return labels 