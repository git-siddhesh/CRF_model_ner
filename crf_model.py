import numpy as np
from typing import List, Dict, Tuple
import math

class CRF:
    def __init__(self, num_states: int, num_features: int, l2_reg: float = 0.01):
        """
        Initialize CRF model with improved initialization
        """
        self.num_states = num_states
        self.num_features = num_features
        self.l2_reg = l2_reg
        
        # Initialize weights with improved Xavier/Glorot initialization
        scale = np.sqrt(6.0 / (num_features + num_states))  # Using 6 instead of 2 for better spread
        self.state_weights = np.random.uniform(-scale, scale, (num_states, num_features))
        self.transition_weights = np.random.uniform(-scale, scale, (num_states, num_states))
        
        # Add bias terms
        self.state_bias = np.zeros(num_states)
        self.transition_bias = np.zeros((num_states, num_states))
        
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
        self.transition_m = np.zeros_like(self.transition_weights)
        self.transition_v = np.zeros_like(self.transition_weights)
        self.state_bias_m = np.zeros_like(self.state_bias)
        self.state_bias_v = np.zeros_like(self.state_bias)
        self.transition_bias_m = np.zeros_like(self.transition_bias)
        self.transition_bias_v = np.zeros_like(self.transition_bias)
        
        # Initialize iteration counter for Adam
        self.t = 0
        
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
    
    def _compute_transition_features(self) -> np.ndarray:
        """
        Compute transition feature scores with bias terms
        """
        return np.clip(self.transition_weights + self.transition_bias, -5, 5)
    
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
        
        Args:
            state_features: Matrix of shape (sequence_length, num_states)
            transition_features: Matrix of shape (num_states, num_states)
            
        Returns:
            Tuple of (log partition function, forward messages)
        """
        sequence_length = state_features.shape[0]
        alpha = np.zeros((sequence_length, self.num_states))
        
        # Initialize first position
        alpha[0] = state_features[0]
        
        # Forward pass in log space
        for t in range(1, sequence_length):
            for j in range(self.num_states):
                # Compute log sum exp of previous states
                scores = alpha[t-1] + transition_features[:, j]
                alpha[t, j] = self._log_sum_exp(scores)
            alpha[t] += state_features[t]
            
        # Compute log partition function
        log_partition = self._log_sum_exp(alpha[-1])
        return log_partition, alpha
    
    def _backward_algorithm(self, state_features: np.ndarray, transition_features: np.ndarray) -> np.ndarray:
        """
        Backward algorithm for computing backward messages in log space
        
        Args:
            state_features: Matrix of shape (sequence_length, num_states)
            transition_features: Matrix of shape (num_states, num_states)
            
        Returns:
            Matrix of backward messages
        """
        sequence_length = state_features.shape[0]
        beta = np.zeros((sequence_length, self.num_states))
        
        # Initialize last position
        beta[-1] = 0.0
        
        # Backward pass in log space
        for t in range(sequence_length-2, -1, -1):
            for i in range(self.num_states):
                # Compute log sum exp of next states
                scores = beta[t+1] + transition_features[i, :] + state_features[t+1]
                beta[t, i] = self._log_sum_exp(scores)
                
        return beta
    
    def _compute_marginals(self, alpha: np.ndarray, beta: np.ndarray, log_partition: float) -> np.ndarray:
        """
        Compute marginal probabilities in log space
        
        Args:
            alpha: Forward messages
            beta: Backward messages
            log_partition: Log of partition function
            
        Returns:
            Matrix of marginal probabilities
        """
        log_marginals = alpha + beta - log_partition
        return np.exp(log_marginals)
    
    def _compute_pairwise_marginals(self, alpha: np.ndarray, beta: np.ndarray, 
                                  state_features: np.ndarray, transition_features: np.ndarray,
                                  log_partition: float) -> np.ndarray:
        """
        Compute pairwise marginal probabilities in log space
        
        Args:
            alpha: Forward messages
            beta: Backward messages
            state_features: State feature scores
            transition_features: Transition feature scores
            log_partition: Log of partition function
            
        Returns:
            Matrix of pairwise marginal probabilities
        """
        sequence_length = alpha.shape[0]
        pairwise_marginals = np.zeros((sequence_length-1, self.num_states, self.num_states))
        
        for t in range(sequence_length-1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    log_pairwise = (alpha[t, i] + 
                                  transition_features[i, j] + 
                                  state_features[t+1, j] + 
                                  beta[t+1, j] - 
                                  log_partition)
                    pairwise_marginals[t, i, j] = np.exp(log_pairwise)
                    
        return pairwise_marginals
    
    def _compute_cross_entropy_loss(self, state_features: np.ndarray, transition_features: np.ndarray,
                                  labels: np.ndarray, log_partition: float) -> float:
        """
        Compute cross-entropy loss with better numerical stability
        """
        # Compute log likelihood of true sequence
        sequence_score = np.sum(state_features[range(len(labels)), labels])
        if len(labels) > 1:
            sequence_score += np.sum(transition_features[labels[:-1], labels[1:]])
        
        # Add small epsilon to prevent log(0)
        return -sequence_score + log_partition + 1e-10
    
    def _update_weights_adam(self, gradients: np.ndarray, weights: np.ndarray,
                           m: np.ndarray, v: np.ndarray, learning_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update weights using Adam optimizer with gradient clipping
        """
        self.t += 1
        t = self.t
        
        # Clip gradients before updating
        grad_norm = np.linalg.norm(gradients)
        if grad_norm > 1.0:
            gradients = gradients / grad_norm
        
        # Update biased first moment estimate
        m = self.beta1 * m + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate
        v = self.beta2 * v + (1 - self.beta2) * (gradients ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.beta1 ** t)
        
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - self.beta2 ** t)
        
        # Update weights with gradient clipping
        update = learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        update_norm = np.linalg.norm(update)
        if update_norm > 1.0:
            update = update / update_norm
            
        weights = weights - update
        
        return weights, m, v
    
    def train(self, features_list: List[np.ndarray], labels_list: List[np.ndarray],
              learning_rate: float = 0.0005, num_iterations: int = 300, patience: int = 50,
              min_iterations: int = 150, validation_features: List[np.ndarray] = None,
              validation_labels: List[np.ndarray] = None, batch_size: int = 16):
        """
        Train the CRF model with improved stability and learning
        """
        best_loss = float('inf')
        patience_counter = 0
        best_iteration = 0
        
        # Initialize learning rate schedule with cosine decay
        initial_lr = learning_rate
        min_lr = learning_rate * 0.01
        
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
                    beta = self._backward_algorithm(state_features, transition_features)
                    
                    # Compute gradients
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
                    batch_loss += self._compute_cross_entropy_loss(state_features, transition_features,
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
                
                # Compute learning rate with cosine decay and warmup
                if iteration < 10:
                    current_lr = initial_lr * (iteration + 1) / 10
                else:
                    progress = (iteration - 10) / (num_iterations - 10)
                    current_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
                
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
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Train Loss: {ema_loss:.4f}, LR: {current_lr:.6f}")
                if validation_features is not None and validation_labels is not None:
                    val_loss = self._compute_validation_loss(validation_features, validation_labels)
                    print(f"Validation Loss: {val_loss:.4f}")
            
            # Early stopping check
            if iteration >= min_iterations:
                val_loss = self._compute_validation_loss(validation_features, validation_labels) if validation_features is not None else ema_loss
                
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
            val_loss += self._compute_cross_entropy_loss(state_features, transition_features,
                                                      labels, log_partition)
        return val_loss / len(validation_features)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels for a sequence of observations
        
        Args:
            features: Feature matrix of shape (sequence_length, num_features)
            
        Returns:
            Array of predicted labels
        """
        sequence_length = features.shape[0]
        
        # Compute feature scores
        state_features = self._compute_state_features(features)
        transition_features = self._compute_transition_features()
        
        # Viterbi algorithm
        viterbi = np.zeros((sequence_length, self.num_states))
        backpointer = np.zeros((sequence_length, self.num_states), dtype=int)
        
        # Initialize first position
        viterbi[0] = state_features[0]
        
        # Forward pass
        for t in range(1, sequence_length):
            for j in range(self.num_states):
                scores = viterbi[t-1] + transition_features[:, j]
                viterbi[t, j] = np.max(scores) + state_features[t, j]
                backpointer[t, j] = np.argmax(scores)
        
        # Backtrack
        labels = np.zeros(sequence_length, dtype=int)
        labels[-1] = np.argmax(viterbi[-1])
        
        for t in range(sequence_length-2, -1, -1):
            labels[t] = backpointer[t+1, labels[t+1]]
            
        return labels 