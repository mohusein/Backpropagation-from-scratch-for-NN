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


search_space = [
    Real(   1e-4,  0.3,  prior='log-uniform', name='lr'        ),
    Integer(16,    128,                        name='h1'        ),
    Integer(8,     64,                         name='h2'        ),
    Categorical(['relu','tanh','leaky_relu'],   name='activation'),
    Real(   1e-5,  0.1,  prior='log-uniform', name='lambda_reg'),
    Integer(150,   400,                        name='n_epochs'  ),
]

# ── Objective function ────────────────────────────────────────────────────────
bo_call_count = [0]
bo_val_accs   = []

@use_named_args(search_space)
def objective(lr, h1, h2, activation, lambda_reg, n_epochs):
    bo_call_count[0] += 1
    nn = NeuralNetwork(
        layer_sizes=[30, int(h1), int(h2), 1],
        activation=activation,
        lambda_reg=float(lambda_reg),
        seed=42
    )
    nn.fit(X_train, y_train,
           lr=float(lr),
           n_epochs=int(n_epochs),
           batch_size=32,
           verbose=False)
    val_acc = nn.score(X_val, y_val)
    bo_val_accs.append(val_acc)
    if bo_call_count[0] % 5 == 0:
        print(f"  Call {bo_call_count[0]:2d} | val_acc = {val_acc:.4f} | "
              f"lr={lr:.5f}, H=[{int(h1)},{int(h2)}], act={activation}")
    return -val_acc   # minimize negative accuracy

# ── Run Bayesian Optimization ─────────────────────────────────────────────────
print("Starting Bayesian Optimization...")
print(f"Search space: lr, h1, h2, activation, lambda_reg, n_epochs")
print("-" * 65)

bo_result = gp_minimize(
    objective,
    search_space,
    n_calls         = 30,
    n_initial_points= 5,
    acq_func        = 'EI',
    random_state    = 42,
    verbose         = False
)

print("-" * 65)
print(f"\nBest validation accuracy : {-bo_result.fun:.4f}")

best = {dim.name: val for dim, val in zip(search_space, bo_result.x)}
print("\nOptimal hyperparameters:")
for k, v in best.items():
    print(f"  {k:12s} : {v}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

calls = list(range(1, len(bo_val_accs) + 1))
running_best = np.maximum.accumulate(bo_val_accs)

axes[0].plot(calls, bo_val_accs, color='#6366f1', lw=1.5, alpha=0.7,
             marker='o', markersize=4, label='Val accuracy')
axes[0].plot(calls, running_best, color='#22c55e', lw=2.0, ls='--', label='Running best')
axes[0].axhline(y=running_best[-1], color='#f59e0b', lw=1, ls=':', alpha=0.6)
axes[0].set_title('BO Convergence', fontsize=12, color='#e2e8f0', pad=8)
axes[0].set_xlabel('Evaluation call')
axes[0].set_ylabel('Validation accuracy')
axes[0].legend(fontsize=9, facecolor='#0d1117', edgecolor='#1e2d5e', labelcolor='#e2e8f0')
axes[0].grid(True)

# Distribution of val accuracies across BO calls
axes[1].hist(bo_val_accs, bins=12, color='#3b82f6', edgecolor='#1e2d5e',
             alpha=0.85, linewidth=0.7)
axes[1].axvline(x=running_best[-1], color='#22c55e', lw=2, ls='--',
                label=f'Best: {running_best[-1]:.4f}')
axes[1].set_title('Val Accuracy Distribution', fontsize=12, color='#e2e8f0', pad=8)
axes[1].set_xlabel('Validation accuracy')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=9, facecolor='#0d1117', edgecolor='#1e2d5e', labelcolor='#e2e8f0')
axes[1].grid(True)

plt.suptitle('Bayesian Optimization Summary', color='#e2e8f0', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

best_lr     = float(best['lr'])
best_h1     = int(best['h1'])
best_h2     = int(best['h2'])
best_act    = best['activation']
best_lam    = float(best['lambda_reg'])
best_epochs = int(best['n_epochs'])

# Sub-model configurations
sub_configs = [
    {
        'name'      : 'BO-Optimal',
        'layers'    : [30, best_h1, best_h2, 1],
        'activation': best_act,
        'lambda_reg': best_lam,
        'lr'        : best_lr,
        'epochs'    : best_epochs,
        'seed'      : 42,
    },
    {
        'name'      : 'Wide-H1',
        'layers'    : [30, int(best_h1 * 1.5), best_h2, 1],
        'activation': best_act,
        'lambda_reg': best_lam,
        'lr'        : best_lr,
        'epochs'    : best_epochs,
        'seed'      : 7,
    },
    {
        'name'      : 'Deep-3L',
        'layers'    : [30, best_h1, best_h2, max(8, best_h2 // 2), 1],
        'activation': best_act,
        'lambda_reg': best_lam * 1.5,
        'lr'        : best_lr * 0.8,
        'epochs'    : best_epochs + 50,
        'seed'      : 13,
    },
    {
        'name'      : 'Tanh-Act',
        'layers'    : [30, best_h1, best_h2, 1],
        'activation': 'tanh',
        'lambda_reg': best_lam,
        'lr'        : best_lr,
        'epochs'    : best_epochs,
        'seed'      : 21,
    },
    {
        'name'      : 'LeakyReLU-LightReg',
        'layers'    : [30, best_h1, best_h2, 1],
        'activation': 'leaky_relu',
        'lambda_reg': best_lam * 0.5,
        'lr'        : best_lr * 1.2,
        'epochs'    : best_epochs,
        'seed'      : 99,
    },
]

# Train all sub-models
trained_models = []
results        = []

print(f"{'Model':<24} {'Val Acc':>8} {'Test Acc':>9} {'F1':>8} {'AUC':>8}")
print("-" * 62)

for cfg in sub_configs:
    nn = NeuralNetwork(
        layer_sizes=cfg['layers'],
        activation =cfg['activation'],
        lambda_reg =cfg['lambda_reg'],
        seed       =cfg['seed'],
    )
    nn.fit(
        X_train_full, y_train_full,
        lr        = cfg['lr'],
        n_epochs  = cfg['epochs'],
        batch_size= 32,
        verbose   = False,
    )
    y_pred      = nn.predict(X_test)
    y_proba     = nn.predict_proba(X_test)
    val_acc     = nn.score(X_val, y_val)
    test_acc    = accuracy_score(y_test,  y_pred)
    f1          = f1_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_proba)

    trained_models.append(nn)
    results.append({
        'Model'     : cfg['name'],
        'Arch'      : str(cfg['layers']),
        'Act'       : cfg['activation'],
        'LR'        : f"{cfg['lr']:.5f}",
        'Val Acc'   : val_acc,
        'Test Acc'  : test_acc,
        'F1'        : f1,
        'AUC'       : auc,
    })
    print(f"{cfg['name']:<24} {val_acc:>8.4f} {test_acc:>9.4f} {f1:>8.4f} {auc:>8.4f}")

print("-" * 62)
df_results = pd.DataFrame(results)
best_model_idx = df_results['Test Acc'].idxmax()
print(f"\nBest model on test set: {df_results.loc[best_model_idx, 'Model']}")
print(f"Test accuracy          : {df_results.loc[best_model_idx, 'Test Acc']:.4f}")


palette = ['#3b82f6', '#10b981', '#f59e0b', '#a78bfa', '#f472b6']

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel 1: Training loss curves ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for i, (nn, row) in enumerate(zip(trained_models, results)):
    ax1.plot(nn.loss_history, color=palette[i], lw=1.8, alpha=0.85,
             label=row['Model'])
ax1.set_title('Training Loss Curves (all sub-models)', color='#e2e8f0', fontsize=11, pad=7)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('BCE + L2 Loss')
ax1.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1e2d5e', labelcolor='#e2e8f0',
           loc='upper right')
ax1.grid(True)

# ── Panel 2: Test accuracy bar chart ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
names = [r['Model'] for r in results]
accs  = [r['Test Acc'] for r in results]
bars  = ax2.barh(names, accs, color=palette, edgecolor='#1e2d5e', linewidth=0.7, height=0.55)
ax2.set_xlim(0.88, 1.01)
ax2.set_title('Test Accuracy', color='#e2e8f0', fontsize=11, pad=7)
ax2.set_xlabel('Accuracy')
for bar, acc in zip(bars, accs):
    ax2.text(acc + 0.001, bar.get_y() + bar.get_height() / 2,
             f'{acc:.4f}', va='center', fontsize=8, color='#e2e8f0')
ax2.grid(True, axis='x')
ax2.tick_params(labelsize=8)

# ── Panel 3: Confusion matrix for best model ─────────────────────────────────
ax3 = fig.add_subplot(gs[1, :2])
best_nn   = trained_models[best_model_idx]
y_pred_bm = best_nn.predict(X_test)
cm        = confusion_matrix(y_test, y_pred_bm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            linewidths=1, linecolor='#080c14',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'],
            cbar_kws={'shrink': 0.75},
            annot_kws={'size': 13, 'weight': 'bold', 'color': '#e2e8f0'})
ax3.set_title(f'Confusion Matrix: {results[best_model_idx]["Model"]}',
              color='#e2e8f0', fontsize=11, pad=7)
ax3.set_xlabel('Predicted label'); ax3.set_ylabel('True label')
ax3.tick_params(labelsize=9)

# ── Panel 4: ROC curves ───────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
for i, (nn, row) in enumerate(zip(trained_models, results)):
    fpr, tpr, _ = roc_curve(y_test, nn.predict_proba(X_test))
    ax4.plot(fpr, tpr, color=palette[i], lw=1.8, alpha=0.85,
             label=f"{row['Model']} ({row['AUC']:.3f})")
ax4.plot([0, 1], [0, 1], 'w--', lw=0.8, alpha=0.4)
ax4.set_title('ROC Curves', color='#e2e8f0', fontsize=11, pad=7)
ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR')
ax4.legend(fontsize=7, facecolor='#0d1117', edgecolor='#1e2d5e', labelcolor='#e2e8f0')
ax4.grid(True)

# ── Panel 5: F1 vs AUC scatter ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, :2])
f1s  = [r['F1']  for r in results]
aucs = [r['AUC'] for r in results]
sc   = ax5.scatter(f1s, aucs, c=palette[:len(results)], s=120,
                   edgecolors='#e2e8f0', linewidths=0.6, zorder=3)
for i, row in enumerate(results):
    ax5.annotate(row['Model'], (f1s[i], aucs[i]),
                 textcoords='offset points', xytext=(8, 3),
                 fontsize=8, color='#94a3b8')
ax5.set_title('F1 Score vs AUC-ROC', color='#e2e8f0', fontsize=11, pad=7)
ax5.set_xlabel('F1 Score'); ax5.set_ylabel('AUC-ROC')
ax5.grid(True)

# ── Panel 6: Results table ────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
col_labels = ['Model', 'Act', 'Test Acc', 'F1', 'AUC']
table_data = [
    [r['Model'], r['Act'], f"{r['Test Acc']:.4f}", f"{r['F1']:.4f}", f"{r['AUC']:.4f}"]
    for r in results
]
tbl = ax6.table(
    cellText   = table_data,
    colLabels  = col_labels,
    cellLoc    = 'center',
    loc        = 'center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)
for (row_i, col_i), cell in tbl.get_celld().items():
    cell.set_facecolor('#0d1420' if row_i % 2 == 0 else '#080c14')
    cell.set_edgecolor('#1e2d5e')
    cell.set_text_props(color='#e2e8f0')
ax6.set_title('Results Summary', color='#e2e8f0', fontsize=10, pad=7)

fig.patch.set_facecolor('#080c14')
plt.suptitle('Sub-Model Evaluation Dashboard', color='#e2e8f0', fontsize=14, y=1.01)
plt.show()

# Classification report for best model
print(f"\nClassification Report: {results[best_model_idx]['Model']}")
print("=" * 55)
print(classification_report(y_test, y_pred_bm,
                             target_names=['Malignant', 'Benign']))






