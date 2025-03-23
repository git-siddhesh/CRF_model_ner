import numpy as np
from typing import List, Dict, Tuple
import math

class CRF:
    def __init__(self, num_states: int, num_features: int, l2_reg: float = 0.01):
        """
        Initialize CRF model with second-order transitions
        """
        self.num_states = num_states
        self.num_features = num_features
        self.l2_reg = l2_reg
        
        # Initialize label dictionary
        self.label_dict = None  # Will be set during training
        
        # Initialize weights with improved Xavier/Glorot initialization
        scale = np.sqrt(6.0 / (num_features + num_states))
        self.state_weights = np.random.uniform(-scale, scale, (num_states, num_features))
        
        # First-order transitions
        self.transition_weights = np.random.uniform(-scale, scale, (num_states, num_states))
        
        # Second-order transitions
        self.second_order_weights = np.random.uniform(-scale, scale, (num_states, num_states, num_states))
        
        # Add bias terms
        self.state_bias = np.zeros(num_states)
        self.transition_bias = np.zeros((num_states, num_states))
        self.second_order_bias = np.zeros((num_states, num_states, num_states))
        
        # Initialize Adam optimizer parameters for second-order weights
        self.second_order_m = np.zeros_like(self.second_order_weights)
        self.second_order_v = np.zeros_like(self.second_order_weights)
        self.second_order_bias_m = np.zeros_like(self.second_order_bias)
        self.second_order_bias_v = np.zeros_like(self.second_order_bias)
        
        # Initialize feature templates
        self.feature_templates = self._initialize_feature_templates()
        
        # Initialize feature statistics for normalization
        self.feature_mean = None
        self.feature_std = None
        
        # Initialize best model weights for early stopping
        self.best_state_weights = None
        self.best_transition_weights = None
        self.best_state_bias = None
        self.best_transition_bias = None
        
        # Initialize Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
        # Initialize Adam moments and velocities for weights and biases
        self.state_m = np.zeros_like(self.state_weights)
        self.state_v = np.zeros_like(self.state_weights)
        self.state_bias_m = np.zeros_like(self.state_bias)
        self.state_bias_v = np.zeros_like(self.state_bias)
        self.transition_m = np.zeros_like(self.transition_weights)
        self.transition_v = np.zeros_like(self.transition_weights)
        self.transition_bias_m = np.zeros_like(self.transition_bias)
        self.transition_bias_v = np.zeros_like(self.transition_bias)
        
        # Initialize iteration counter for Adam
        self.t = 0
        
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
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to have zero mean and unit variance
        
        Args:
            features: Feature matrix of shape (sequence_length, num_features)
            
        Returns:
            Normalized feature matrix
        """
        if self.feature_mean is None:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0)
            self.feature_std[self.feature_std == 0] = 1  # Avoid division by zero
            
        normalized = (features - self.feature_mean) / self.feature_std
        # Clip normalized values to prevent extreme values
        return np.clip(normalized, -10, 10)
    
    def _clip_gradients(self, gradients: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        """
        Clip gradients to prevent exploding gradients
        
        Args:
            gradients: Gradient matrix
            max_norm: Maximum norm for gradient clipping
            
        Returns:
            Clipped gradients
        """
        norm = np.linalg.norm(gradients)
        if norm > max_norm:
            gradients = gradients * (max_norm / norm)
        return gradients
    
    def _compute_state_features(self, features: np.ndarray) -> np.ndarray:
        """
        Compute state feature scores with bias terms
        """
        normalized_features = self._normalize_features(features)
        scores = np.dot(normalized_features, self.state_weights.T) + self.state_bias
        return np.clip(scores, -5, 5)  # Reduced clipping range
    
    def _compute_transition_features(self, prev_state=None, prev_prev_state=None) -> np.ndarray:
        """
        Compute transition feature scores with second-order transitions
        """
        if prev_prev_state is None or prev_state is None:
            return np.clip(self.transition_weights + self.transition_bias, -5, 5)
        
        # Combine first and second-order transitions
        first_order = self.transition_weights + self.transition_bias
        second_order = self.second_order_weights[prev_prev_state, prev_state] + self.second_order_bias[prev_prev_state, prev_state]
        
        return np.clip(first_order + second_order, -5, 5)
    
    def _log_sum_exp(self, x: np.ndarray) -> float:
        """
        Compute log(sum(exp(x))) in a numerically stable way
        
        Args:
            x: Array of values
            
        Returns:
            Log of sum of exponentials
        """
        max_x = np.max(x)
        return max_x + np.log(np.sum(np.exp(x - max_x)))
    
    def _forward_algorithm(self, state_features: np.ndarray, transition_features: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Forward algorithm for computing partition function and forward messages in log space
        """
        sequence_length = state_features.shape[0]
        alpha = np.zeros((sequence_length, self.num_states))
        
        # Initialize first position
        alpha[0] = state_features[0]
        
        # Forward pass in log space
        for t in range(1, sequence_length):
            for j in range(self.num_states):
                # Add state features after transition scores
                scores = alpha[t-1] + transition_features[:, j] + state_features[t, j]
                alpha[t, j] = self._log_sum_exp(scores)
        
        # Compute log partition function
        log_partition = self._log_sum_exp(alpha[-1])
        return log_partition, alpha
    
    def _backward_algorithm(self, state_features: np.ndarray, transition_features: np.ndarray) -> np.ndarray:
        """
        Backward algorithm for computing backward messages in log space
        """
        sequence_length = state_features.shape[0]
        beta = np.zeros((sequence_length, self.num_states))
        
        # Initialize last position
        beta[-1] = 0.0
        
        # Backward pass in log space
        for t in range(sequence_length-2, -1, -1):
            for i in range(self.num_states):
                scores = beta[t+1] + transition_features[i, :] + state_features[t+1]
                beta[t, i] = self._log_sum_exp(scores)
        
        return beta
    
    def _compute_marginals(self, alpha: np.ndarray, beta: np.ndarray, log_partition: float) -> np.ndarray:
        """
        Compute marginal probabilities in log space with numerical stability
        """
        # Compute log marginals
        log_marginals = alpha + beta - log_partition
        
        # Convert to probabilities with numerical stability
        max_vals = np.max(log_marginals, axis=1, keepdims=True)
        exp_vals = np.exp(log_marginals - max_vals)
        marginals = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        return marginals
    
    def _compute_pairwise_marginals(self, alpha: np.ndarray, beta: np.ndarray, 
                                  state_features: np.ndarray, transition_features: np.ndarray,
                                  log_partition: float) -> np.ndarray:
        """
        Compute pairwise marginal probabilities with numerical stability
        """
        sequence_length = alpha.shape[0]
        pairwise_marginals = np.zeros((sequence_length-1, self.num_states, self.num_states))
        
        for t in range(sequence_length-1):
            # Compute log pairwise marginals
            log_pairwise = (alpha[t, :, np.newaxis] + 
                           transition_features + 
                           state_features[t+1] + 
                           beta[t+1] - 
                           log_partition)
            
            # Convert to probabilities with numerical stability
            max_val = np.max(log_pairwise)
            exp_vals = np.exp(log_pairwise - max_val)
            pairwise_marginals[t] = exp_vals / np.sum(exp_vals)
        
        return pairwise_marginals
    
    def _compute_crf_loss(self, state_features: np.ndarray, transition_features: np.ndarray,
                        labels: np.ndarray, log_partition: float) -> float:
        """
        Compute CRF loss with proper scaling
        """
        sequence_length = len(labels)
        
        # Scale down features to prevent explosion
        state_features = np.clip(state_features, -10, 10)
        transition_features = np.clip(transition_features, -10, 10)
        
        # Compute unary potentials (state scores)
        unary_score = np.sum(state_features[range(sequence_length), labels])
        
        # Compute pairwise potentials (transition scores)
        pairwise_score = 0
        if sequence_length > 1:
            for t in range(sequence_length - 1):
                current_label = labels[t]
                next_label = labels[t + 1]
                
                # Add transition score
                transition_score = transition_features[current_label, next_label]
                
                # Apply BIO constraints penalties (scaled down)
                if self._violates_bio_constraints(current_label, next_label):
                    transition_score -= 10.0  # Reduced penalty
                    
                pairwise_score += transition_score
        
        # Total score for the correct sequence
        sequence_score = unary_score + pairwise_score
        
        # Compute loss with label smoothing (scaled)
        epsilon = 0.1  # Label smoothing factor
        smooth_loss = (1 - epsilon) * (-sequence_score + log_partition) + \
                     epsilon * self._compute_label_smoothing_loss(state_features, labels)
        
        # Scale down the final loss
        return smooth_loss / sequence_length  # Normalize by sequence length

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

    def _compute_label_smoothing_loss(self, state_features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute label smoothing loss to prevent overconfident predictions
        """
        uniform_distribution = np.ones(self.num_states) / self.num_states
        kl_divergence = 0.0
        
        for t in range(len(labels)):
            logits = state_features[t]
            probs = np.exp(logits - self._log_sum_exp(logits))
            kl_divergence += np.sum(uniform_distribution * np.log(uniform_distribution / (probs + 1e-10)))
        
        return kl_divergence / len(labels)

    def _update_weights_adam(self, gradients: np.ndarray, weights: np.ndarray, 
                              m: np.ndarray, v: np.ndarray, learning_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        v = self.beta2 * v + (1 - self.beta2) * np.square(gradients)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.beta1**self.t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.beta2**self.t)
        
        # Update weights
        weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return weights, m, v

    def train(self, features_list: List[np.ndarray], labels_list: List[np.ndarray],
              learning_rate: float = 0.0005, num_iterations: int = 300, patience: int = 50,
              min_iterations: int = 150, validation_features: List[np.ndarray] = None,
              validation_labels: List[np.ndarray] = None, batch_size: int = 16):
        """
        Train the CRF model with improved stability and learning
        
        Returns:
            Dictionary containing training history
        """
        best_loss = float('inf')
        patience_counter = 0
        best_iteration = 0
        
        # Initialize history tracking
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Store label dictionary from data_loader
        self.label_dict = {
            'O': 0,
            'B-art': 1, 'I-art': 2,
            'B-eve': 3, 'I-eve': 4,
            'B-geo': 5, 'I-geo': 6,
            'B-gpe': 7, 'I-gpe': 8,
            'B-nat': 9, 'I-nat': 10,
            'B-org': 11, 'I-org': 12,
            'B-per': 13, 'I-per': 14,
            'B-tim': 15, 'I-tim': 16
        }
        
        # Create reverse mapping for integer labels
        int_to_bio = {
            0: 'O',
            1: 'B-art', 2: 'I-art',
            3: 'B-eve', 4: 'I-eve', 
            5: 'B-geo', 6: 'I-geo',
            7: 'B-gpe', 8: 'I-gpe',
            9: 'B-nat', 10: 'I-nat',
            11: 'B-org', 12: 'I-org',
            13: 'B-per', 14: 'I-per',
            15: 'B-tim', 16: 'I-tim'
        }
        
        print("Label dictionary:", self.label_dict)
        
        # Convert input labels directly - they should already be correct indices
        labels_list = [np.array(labels) for labels in labels_list]
        if validation_labels is not None:
            validation_labels = [np.array(labels) for labels in validation_labels]
        
        # Initialize learning rate schedule with cosine decay
        initial_lr = learning_rate
        min_lr = learning_rate * 0.01
        warmup_steps = 5
        
        # Initialize exponential moving average for loss
        ema_loss = None
        ema_alpha = 0.1
        
        # Initialize gradient accumulation
        grad_accum_steps = 4  # Accumulate gradients over 4 steps
        
        for iteration in range(num_iterations):
            total_loss = 0
            
            # Shuffle training data
            indices = np.random.permutation(len(features_list))
            features_list = [features_list[i] for i in indices]
            labels_list = [labels_list[i] for i in indices]
            
            # Process data in mini-batches
            for batch_start in range(0, len(features_list), batch_size):
                batch_end = min(batch_start + batch_size, len(features_list))
                batch_features = features_list[batch_start:batch_end]
                batch_labels = labels_list[batch_start:batch_end]
                
                # Reset gradients for this batch
                state_gradients = np.zeros_like(self.state_weights)
                transition_gradients = np.zeros_like(self.transition_weights)
                state_bias_gradients = np.zeros_like(self.state_bias)
                transition_bias_gradients = np.zeros_like(self.transition_bias)
                
                batch_loss = 0
                for features, labels in zip(batch_features, batch_labels):
                    # Forward pass
                    state_features = self._compute_state_features(features)
                    transition_features = self._compute_transition_features()
                    log_partition, alpha = self._forward_algorithm(state_features, transition_features)
                    
                    # Compute CRF loss with BIO constraints
                    batch_loss += self._compute_crf_loss(state_features, transition_features,
                                                       labels, log_partition)
                    
                    # Compute gradients
                    beta = self._backward_algorithm(state_features, transition_features)
                    marginals = self._compute_marginals(alpha, beta, log_partition)
                    pairwise_marginals = self._compute_pairwise_marginals(alpha, beta, state_features, 
                                                                        transition_features, log_partition)
                    
                    # Compute sequence gradients
                    seq_state_gradients = np.zeros_like(self.state_weights)
                    seq_transition_gradients = np.zeros_like(self.transition_weights)
                    seq_state_bias_gradients = np.zeros_like(self.state_bias)
                    seq_transition_bias_gradients = np.zeros_like(self.transition_bias)
                    
                    # Update feature weights and biases
                    for t in range(len(labels)):
                        seq_state_gradients[labels[t]] += features[t]
                        seq_state_gradients -= marginals[t, :, np.newaxis] * features[t]
                        seq_state_bias_gradients[labels[t]] += 1
                        seq_state_bias_gradients -= marginals[t]
                    
                    # Update transition weights and biases
                    for t in range(len(labels)-1):
                        seq_transition_gradients[labels[t], labels[t+1]] += 1
                        seq_transition_gradients -= pairwise_marginals[t]
                        seq_transition_bias_gradients[labels[t], labels[t+1]] += 1
                        seq_transition_bias_gradients -= pairwise_marginals[t]
                    
                    # Accumulate gradients
                    state_gradients += seq_state_gradients / grad_accum_steps
                    transition_gradients += seq_transition_gradients / grad_accum_steps
                    state_bias_gradients += seq_state_bias_gradients / grad_accum_steps
                    transition_bias_gradients += seq_transition_bias_gradients / grad_accum_steps
                    
                    # Compute loss
                    batch_loss += self._compute_crf_loss(state_features, transition_features,
                                                        labels, log_partition)
                
                # Average gradients and loss
                batch_size_actual = len(batch_features)
                state_gradients /= batch_size_actual
                transition_gradients /= batch_size_actual
                state_bias_gradients /= batch_size_actual
                transition_bias_gradients /= batch_size_actual
                batch_loss /= batch_size_actual
                
                # Add L2 regularization
                state_gradients += self.l2_reg * self.state_weights
                transition_gradients += self.l2_reg * self.transition_weights
                
                # Compute learning rate with warmup and cosine decay
                if iteration < warmup_steps:
                    current_lr = initial_lr * (iteration + 1) / warmup_steps
                else:
                    progress = (iteration - warmup_steps) / (num_iterations - warmup_steps)
                    current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
                
                # Update weights with scaled learning rate
                current_lr *= 0.1  # Scale down learning rate for stability
                
                # Update weights and biases using Adam
                self.state_weights, self.state_m, self.state_v = self._update_weights_adam(
                    state_gradients, self.state_weights, self.state_m, self.state_v, current_lr)
                self.transition_weights, self.transition_m, self.transition_v = self._update_weights_adam(
                    transition_gradients, self.transition_weights, self.transition_m, self.transition_v, current_lr)
                self.state_bias, self.state_bias_m, self.state_bias_v = self._update_weights_adam(
                    state_bias_gradients, self.state_bias, self.state_bias_m, self.state_bias_v, current_lr)
                self.transition_bias, self.transition_bias_m, self.transition_bias_v = self._update_weights_adam(
                    transition_bias_gradients, self.transition_bias, self.transition_bias_m, self.transition_bias_v, current_lr)
                
                # Clip weights and biases
                self.state_weights = np.clip(self.state_weights, -5, 5)
                self.transition_weights = np.clip(self.transition_weights, -5, 5)
                self.state_bias = np.clip(self.state_bias, -5, 5)
                self.transition_bias = np.clip(self.transition_bias, -5, 5)
                
                total_loss += batch_loss
            
            avg_loss = total_loss / (len(features_list) // batch_size)
            
            # Update exponential moving average of loss
            if ema_loss is None:
                ema_loss = avg_loss
            else:
                ema_loss = ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss
            
            # Compute validation loss
            if validation_features is not None and validation_labels is not None:
                val_loss = self._compute_validation_loss(validation_features, validation_labels)
            else:
                val_loss = ema_loss
            
            # Store history
            history['train_loss'].append(ema_loss)
            history['val_loss'].append(val_loss)
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Train Loss: {ema_loss:.4f}, LR: {current_lr:.6f}")
                if validation_features is not None and validation_labels is not None:
                    print(f"Validation Loss: {val_loss:.4f}")
            
            # Early stopping check
            if iteration >= min_iterations:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_iteration = iteration
                    # Save best model
                    self.best_state_weights = self.state_weights.copy()
                    self.best_transition_weights = self.transition_weights.copy()
                    self.best_state_bias = self.state_bias.copy()
                    self.best_transition_bias = self.transition_bias.copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {iteration}")
                        print(f"Best model was at iteration {best_iteration}")
                        # Restore best model
                        self.state_weights = self.best_state_weights
                        self.transition_weights = self.best_transition_weights
                        self.state_bias = self.best_state_bias
                        self.transition_bias = self.best_transition_bias
                        break
        
        return history
    
    def _compute_validation_loss(self, validation_features: List[np.ndarray],
                               validation_labels: List[np.ndarray]) -> float:
        """
        Compute validation loss
        """
        val_loss = 0
        for features, labels in zip(validation_features, validation_labels):
            state_features = self._compute_state_features(features)
            transition_features = self._compute_transition_features()
            log_partition, _ = self._forward_algorithm(state_features, transition_features)
            val_loss += self._compute_crf_loss(state_features, transition_features,
                                              labels, log_partition)
        return val_loss / len(validation_features)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using Viterbi algorithm with second-order transitions
        """
        sequence_length = features.shape[0]
        
        # Compute feature scores
        state_features = self._compute_state_features(features)
        
        # Initialize Viterbi variables for second-order transitions
        viterbi = np.zeros((sequence_length, self.num_states, self.num_states))
        backpointer = np.zeros((sequence_length, self.num_states, self.num_states), dtype=int)
        
        # Initialize first position
        viterbi[0] = state_features[0][:, np.newaxis] + self.transition_weights
        
        # Forward pass with second-order transitions
        for t in range(2, sequence_length):
            for k in range(self.num_states):  # current state
                for j in range(self.num_states):  # previous state
                    # Include both first and second-order transitions
                    scores = (viterbi[t-1, :, j] + 
                            self.transition_weights[:, k] +
                            self.second_order_weights[:, j, k] +
                            state_features[t, k])
                    viterbi[t, j, k] = np.max(scores)
                    backpointer[t, j, k] = np.argmax(scores)
        
        # Backtrack
        labels = np.zeros(sequence_length, dtype=int)
        
        # Find best last two states
        last_scores = viterbi[-1]
        labels[-2], labels[-1] = np.unravel_index(np.argmax(last_scores), last_scores.shape)
        
        # Backtrack remaining states
        for t in range(sequence_length-3, -1, -1):
            labels[t] = backpointer[t+2, labels[t+1], labels[t+2]]
        
        return labels 