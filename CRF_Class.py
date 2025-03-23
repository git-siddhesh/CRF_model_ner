import numpy as np

import scipy


from scipy import optimize
from collections import defaultdict, Counter

class CRF:
    def __init__(self, feature_func=None, max_iterations=100):
        """
        Initialize the CRF model.
        
        Args:
            feature_func: A function that extracts features from (x, y, y_prev, i) where:
                          x is the input sequence
                          y is the current label
                          y_prev is the previous label
                          i is the position in the sequence
            max_iterations: Maximum number of iterations for L-BFGS optimization
        """
        self.feature_func = feature_func
        self.max_iterations = max_iterations
        self.weights = None
        self.feature_map = {}
        self.labels = set()
        self.num_features = 0
    
    def _extract_features(self, x_seq, y_seq):
        """
        Extract features from the input sequence x_seq and label sequence y_seq.
        
        Args:
            x_seq: Input sequence
            y_seq: Label sequence
            
        Returns:
            A dictionary of feature counts
        """
        features = defaultdict(float)
        y_prev = "<START>"
        
        for i, (x, y) in enumerate(zip(x_seq, y_seq)):
            for feature in self.feature_func(x_seq, y, y_prev, i):
                features[feature] += 1
            y_prev = y
            
        return features
    
    def _extract_feature_vector(self, features):
        """
        Convert feature dictionary to a feature vector.
        
        Args:
            features: Dictionary of feature counts
            
        Returns:
            Feature vector (numpy array)
        """
        vec = np.zeros(self.num_features)
        for feature, count in features.items():
            if feature in self.feature_map:
                vec[self.feature_map[feature]] = count
        return vec
    
    def _build_feature_map(self, X, Y):
        """
        Build a mapping from features to indices.
        
        Args:
            X: List of input sequences
            Y: List of label sequences
        """
        feature_set = set()
        self.labels = set()
        
        for x_seq, y_seq in zip(X, Y):
            y_prev = "<START>"
            for i, (x, y) in enumerate(zip(x_seq, y_seq)):
                self.labels.add(y)
                for feature in self.feature_func(x_seq, y, y_prev, i):
                    feature_set.add(feature)
                y_prev = y
        
        self.feature_map = {feature: idx for idx, feature in enumerate(feature_set)}
        self.num_features = len(self.feature_map)
        
        print(f"Number of features: {self.num_features}")
        print(f"Number of labels: {len(self.labels)}")
    
    def _compute_log_potential(self, x_seq, y_seq):
        """
        Compute the log potential of a sequence pair.
        
        Args:
            x_seq: Input sequence
            y_seq: Label sequence
            
        Returns:
            Log potential (float)
        """
        features = self._extract_features(x_seq, y_seq)
        vec = self._extract_feature_vector(features)
        return np.dot(self.weights, vec)
    
    def _compute_forward_values(self, x_seq):
        """
        Compute forward values for the forward-backward algorithm.
        
        Args:
            x_seq: Input sequence
            
        Returns:
            Forward values matrix (numpy array)
        """
        n = len(x_seq)
        labels = list(self.labels)
        m = len(labels)
        
        # Initialize forward values
        forward = np.zeros((n, m))
        
        # Base case: first position
        y_prev = "<START>"
        for j, y in enumerate(labels):
            for feature in self.feature_func(x_seq, y, y_prev, 0):
                if feature in self.feature_map:
                    forward[0, j] += self.weights[self.feature_map[feature]]
        
        # Forward pass
        for i in range(1, n):
            for j, y in enumerate(labels):
                max_val = float("-inf")
                for k, y_prev in enumerate(labels):
                    val = forward[i-1, k]
                    for feature in self.feature_func(x_seq, y, y_prev, i):
                        if feature in self.feature_map:
                            val += self.weights[self.feature_map[feature]]
                    max_val = max(max_val, val)
                forward[i, j] = max_val
        
        return forward, labels
    
    def _compute_log_likelihood(self, X, Y):
        """
        Compute the log-likelihood of the data.
        
        Args:
            X: List of input sequences
            Y: List of label sequences
            
        Returns:
            Log-likelihood (float)
        """
        log_likelihood = 0
        
        for x_seq, y_seq in zip(X, Y):
            log_likelihood += self._compute_log_potential(x_seq, y_seq)
            
            # Compute partition function
            forward, labels = self._compute_forward_values(x_seq)
            Z = np.logaddexp.reduce(forward[-1, :])
            log_likelihood -= Z
        
        # Add L2 regularization
        log_likelihood -= 0.5 * np.sum(self.weights ** 2)
        
        return log_likelihood
    
    def _compute_gradient(self, X, Y):
        """
        Compute the gradient of the log-likelihood.
        
        Args:
            X: List of input sequences
            Y: List of label sequences
            
        Returns:
            Gradient (numpy array)
        """
        gradient = np.zeros(self.num_features)
        
        # Empirical feature counts
        for x_seq, y_seq in zip(X, Y):
            features = self._extract_features(x_seq, y_seq)
            vec = self._extract_feature_vector(features)
            gradient += vec
        
        # Expected feature counts
        for x_seq in X:
            n = len(x_seq)
            labels = list(self.labels)
            m = len(labels)
            
            # Compute forward values
            forward, labels = self._compute_forward_values(x_seq)
            
            # Compute backward values
            backward = np.zeros((n, m))
            
            # Base case: last position
            backward[-1, :] = 0
            
            # Backward pass
            for i in range(n-2, -1, -1):
                for j, y_prev in enumerate(labels):
                    max_val = float("-inf")
                    for k, y in enumerate(labels):
                        val = backward[i+1, k]
                        for feature in self.feature_func(x_seq, y, y_prev, i+1):
                            if feature in self.feature_map:
                                val += self.weights[self.feature_map[feature]]
                        max_val = max(max_val, val)
                    backward[i, j] = max_val
            
            # Compute partition function
            Z = np.logaddexp.reduce(forward[-1, :])
            
            # Compute expected feature counts
            expected_counts = defaultdict(float)
            
            for i in range(n):
                for j, y in enumerate(labels):
                    for k, y_prev in enumerate(labels) if i > 0 else [("<START>", -1)]:
                        if i > 0:
                            p = forward[i-1, k] + backward[i, j]
                            for feature in self.feature_func(x_seq, y, y_prev, i):
                                if feature in self.feature_map:
                                    p += self.weights[self.feature_map[feature]]
                            p = np.exp(p - Z)
                        else:
                            p = forward[i, j] + backward[i, j]
                            p = np.exp(p - Z)
                        
                        for feature in self.feature_func(x_seq, y, y_prev if i > 0 else "<START>", i):
                            if feature in self.feature_map:
                                expected_counts[feature] += p
            
            # Subtract expected counts from gradient
            for feature, count in expected_counts.items():
                if feature in self.feature_map:
                    gradient[self.feature_map[feature]] -= count
        
        # Add L2 regularization
        gradient -= self.weights
        
        return -gradient  # Negate for minimization
    
    def _objective_function(self, weights, X, Y):
        """
        Objective function for L-BFGS optimization.
        
        Args:
            weights: Model weights
            X: List of input sequences
            Y: List of label sequences
            
        Returns:
            Negative log-likelihood (float)
        """
        self.weights = weights
        return -self._compute_log_likelihood(X, Y)
    
    def _objective_gradient(self, weights, X, Y):
        """
        Gradient of the objective function for L-BFGS optimization.
        
        Args:
            weights: Model weights
            X: List of input sequences
            Y: List of label sequences
            
        Returns:
            Gradient of the objective function (numpy array)
        """
        self.weights = weights
        return self._compute_gradient(X, Y)
    
    def fit(self, X, Y):
        """
        Train the CRF model.
        
        Args:
            X: List of input sequences
            Y: List of label sequences
        """
        # Build feature map
        self._build_feature_map(X, Y)
        
        # Initialize weights
        self.weights = np.zeros(self.num_features)
        
        # Optimize using L-BFGS
        print("Starting optimization...")
        result = optimize.minimize(
            self._objective_function,
            self.weights,
            args=(X, Y),
            method='L-BFGS-B',
            jac=self._objective_gradient,
            options={'maxiter': self.max_iterations, 'disp': True}
        )
        
        self.weights = result.x
        print("Optimization complete.")
        
        return self
    
    def viterbi(self, x_seq):
        """
        Use the Viterbi algorithm to find the most probable label sequence.
        
        Args:
            x_seq: Input sequence
            
        Returns:
            Most probable label sequence
        """
        n = len(x_seq)
        labels = list(self.labels)
        m = len(labels)
        
        # Initialize viterbi and backpointer matrices
        viterbi = np.zeros((n, m))
        backpointer = np.zeros((n, m), dtype=int)
        
        # Base case: first position
        y_prev = "<START>"
        for j, y in enumerate(labels):
            for feature in self.feature_func(x_seq, y, y_prev, 0):
                if feature in self.feature_map:
                    viterbi[0, j] += self.weights[self.feature_map[feature]]
        
        # Viterbi algorithm
        for i in range(1, n):
            for j, y in enumerate(labels):
                max_val = float("-inf")
                max_idx = -1
                for k, y_prev in enumerate(labels):
                    val = viterbi[i-1, k]
                    for feature in self.feature_func(x_seq, y, y_prev, i):
                        if feature in self.feature_map:
                            val += self.weights[self.feature_map[feature]]
                    if val > max_val:
                        max_val = val
                        max_idx = k
                viterbi[i, j] = max_val
                backpointer[i, j] = max_idx
        
        # Find the best path
        best_path = [0] * n
        best_path[n-1] = np.argmax(viterbi[n-1, :])
        
        for i in range(n-1, 0, -1):
            best_path[i-1] = backpointer[i, best_path[i]]
        
        return [labels[i] for i in best_path]
    
    def predict(self, X):
        """
        Predict label sequences for a list of input sequences.
        
        Args:
            X: List of input sequences
            
        Returns:
            List of predicted label sequences
        """
        return [self.viterbi(x_seq) for x_seq in X]

# Example usage:
def word_features(x_seq, y, y_prev, i):
    """
    Extract features from a word sequence for named entity recognition.
    
    Args:
        x_seq: Sequence of words
        y: Current label
        y_prev: Previous label
        i: Position in sequence
    
    Returns:
        List of features
    """
    word = x_seq[i]
    features = [
        f'bias',
        f'word={word.lower()}',
        f'word.lower()={word.lower()}',
        f'word.isupper()={word.isupper()}',
        f'word.istitle()={word.istitle()}',
        f'word.isdigit()={word.isdigit()}',
        f'label={y}',
        f'label-prev={y_prev}',
        f'label-prev-curr={y_prev}|{y}'
    ]
    
    # Add prefix and suffix features
    for length in range(1, 4):
        if len(word) >= length:
            features.append(f'prefix{length}={word[:length].lower()}')
            features.append(f'suffix{length}={word[-length:].lower()}')
    
    # Add context features
    if i > 0: 
        prev_word = x_seq[i-1]
        features.extend([
            f'prev_word={prev_word.lower()}',
            f'prev_word|word={prev_word.lower()}|{word.lower()}'
        ])
    else:
        features.append('BOS')
    
    if i < len(x_seq) - 1:
        next_word = x_seq[i+1]
        features.extend([
            f'next_word={next_word.lower()}',
            f'word|next_word={word.lower()}|{next_word.lower()}'
        ])
    else:
        features.append('EOS')
    
    return features

