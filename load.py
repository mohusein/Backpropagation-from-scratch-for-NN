class NeuralNetwork:
    '''Full-scratch feedforward neural network with backpropagation.'''

    def __init__(self, layer_sizes, activation='relu', lambda_reg=1e-4, seed=42):
        '''
        Parameters
        ----------
        layer_sizes : list[int]
            Neuron counts per layer including input and output.
            Example: [30, 64, 32, 1]
        activation  : str
            Hidden-layer activation. One of relu, leaky_relu, tanh, sigmoid.
        lambda_reg  : float
            L2 regularization coefficient.
        seed        : int
            NumPy random seed for reproducibility.
        '''
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.activation  = activation
        self.lambda_reg  = lambda_reg
        self.n_layers    = len(layer_sizes) - 1
        self.weights     = []
        self.biases      = []
        self.loss_history = []
        self._init_weights()

    # ── weight initialization ─────────────────────────────────────────────────
    def _init_weights(self):
        for i in range(self.n_layers):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            if self.activation in ('relu', 'leaky_relu'):
                scale = np.sqrt(2.0 / fan_in)           # He
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
            W = np.random.randn(fan_in, fan_out) * scale
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    # ── activation functions ──────────────────────────────────────────────────
    def _act(self, z, is_output=False):
        if is_output:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        fn = self.activation
        if fn == 'relu':       return np.maximum(0.0, z)
        if fn == 'leaky_relu': return np.where(z > 0, z, 0.01 * z)
        if fn == 'tanh':       return np.tanh(z)
        if fn == 'sigmoid':    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        raise ValueError(f"Unknown activation: {fn}")

    def _act_deriv(self, a, z, is_output=False):
        if is_output:   return a * (1.0 - a)
        fn = self.activation
        if fn == 'relu':       return (z > 0.0).astype(float)
        if fn == 'leaky_relu': return np.where(z > 0, 1.0, 0.01)
        if fn == 'tanh':       return 1.0 - a ** 2
        if fn == 'sigmoid':    return a * (1.0 - a)

    # ── forward pass ──────────────────────────────────────────────────────────
    def forward(self, X):
        '''Compute output and cache intermediates for backprop.'''
        self._z_cache = []   # pre-activation values
        self._a_cache = [X]  # post-activation values (a[0] = input)
        a = X
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self._z_cache.append(z)
            is_out = (i == self.n_layers - 1)
            a = self._act(z, is_output=is_out)
            self._a_cache.append(a)
        return a  # shape (m, 1)

    # ── loss ──────────────────────────────────────────────────────────────────
    def compute_loss(self, y_pred, y_true):
        m   = y_true.shape[0]
        eps = 1e-12
        bce = -np.mean(
            y_true * np.log(y_pred + eps) +
            (1.0 - y_true) * np.log(1.0 - y_pred + eps)
        )
        l2  = (self.lambda_reg / (2.0 * m)) * sum(np.sum(w ** 2) for w in self.weights)
        return bce + l2

    # ── backward pass ─────────────────────────────────────────────────────────
    def backward(self, y_true, lr):
        '''
        Analytical backpropagation via chain rule.

        For cross-entropy loss with sigmoid output the combined gradient of
        loss w.r.t. the last pre-activation collapses to:
            delta_L = y_pred - y_true

        For hidden layers:
            delta_l = (delta_{l+1} @ W_{l+1}^T) * g_prime(z_l)

        Weight gradient with L2:
            dL/dW_l = (a_{l-1}^T @ delta_l) / m  +  (lambda/m) * W_l
        '''
        m       = y_true.shape[0]
        grads_w = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        # Output layer: simplified delta for BCE + sigmoid
        delta = self._a_cache[-1] - y_true  # (m, 1)

        for i in reversed(range(self.n_layers)):
            grads_w[i] = (
                (self._a_cache[i].T @ delta) / m +
                (self.lambda_reg / m) * self.weights[i]
            )
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)

            if i > 0:
                # Propagate error to previous layer
                delta = (delta @ self.weights[i].T) * \
                        self._act_deriv(self._a_cache[i], self._z_cache[i - 1])

        # SGD weight update
        for i in range(self.n_layers):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i]  -= lr * grads_b[i]

        return grads_w, grads_b

    # ── training loop ─────────────────────────────────────────────────────────
    def fit(self, X, y, lr=0.01, n_epochs=300, batch_size=32, verbose=False):
        y = y.reshape(-1, 1).astype(float)
        m = X.shape[0]
        self.loss_history = []

        for epoch in range(n_epochs):
            idx       = np.random.permutation(m)
            X_shuf, y_shuf = X[idx], y[idx]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, m, batch_size):
                end    = min(start + batch_size, m)
                Xb, yb = X_shuf[start:end], y_shuf[start:end]
                yp     = self.forward(Xb)
                epoch_loss += self.compute_loss(yp, yb)
                self.backward(yb, lr)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            if verbose and epoch % 50 == 0:
                val_acc = self.score(X_val, y_val) if 'X_val' in dir() else 0
                print(f"  Epoch {epoch:4d} | loss = {avg_loss:.5f}")

        return self

    # ── inference helpers ─────────────────────────────────────────────────────
    def predict_proba(self, X):
        return self.forward(X).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


print("NeuralNetwork class defined.")



def numerical_gradient_check(nn, X_check, y_check, eps=1e-5):
    '''
    Compare analytical backprop gradients against numerical finite differences.
    Returns max relative error across all checked parameters.
    '''
    y_check = y_check.reshape(-1, 1).astype(float)

    # Analytical gradients
    nn.forward(X_check)
    grads_analytical, _ = nn.backward(y_check, lr=0.0)  # lr=0 = no update

    max_rel_error = 0.0
    results = []

    for layer_idx in range(nn.n_layers):
        W = nn.weights[layer_idx]
        # Check a random subset of weights to keep runtime low
        check_indices = list(zip(
            np.random.randint(0, W.shape[0], min(6, W.shape[0])),
            np.random.randint(0, W.shape[1], min(6, W.shape[1]))
        ))
        for r, c in check_indices:
            orig = W[r, c]

            W[r, c] = orig + eps
            loss_plus = nn.compute_loss(nn.forward(X_check), y_check)

            W[r, c] = orig - eps
            loss_minus = nn.compute_loss(nn.forward(X_check), y_check)

            W[r, c] = orig  # restore

            num_grad  = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad  = grads_analytical[layer_idx][r, c]
            rel_error = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-12)

            max_rel_error = max(max_rel_error, rel_error)
            results.append({
                'layer': layer_idx,
                'w_idx': f"[{r},{c}]",
                'numerical': f"{num_grad:.6f}",
                'analytical': f"{ana_grad:.6f}",
                'rel_error': f"{rel_error:.2e}",
                'status': 'PASS' if rel_error < 1e-4 else 'FAIL'
            })

    return max_rel_error, pd.DataFrame(results)


# Small network for gradient check
np.random.seed(0)
X_check = X_train[:5]
y_check = y_train[:5]

nn_check = NeuralNetwork([30, 8, 4, 1], activation='relu', lambda_reg=1e-3, seed=7)
nn_check.forward(X_check)  # populate cache

max_err, df_check = numerical_gradient_check(nn_check, X_check, y_check)

print("Gradient Check Results")
print("=" * 60)
print(df_check.to_string(index=False))
print("=" * 60)
status_str = "PASS" if max_err < 1e-4 else "FAIL"
print(f"\nMax relative error : {max_err:.2e}   [{status_str}]")
print("Threshold          : 1e-4")