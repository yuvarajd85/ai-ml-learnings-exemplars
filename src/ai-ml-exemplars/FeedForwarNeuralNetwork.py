"""
Feedforward neural network, built from scratch with plain numpy — no autograd.

Architecture (as specified):
    Input -> H1(16) -> H2(32) -> H3(64) -> H4(32) -> H5(16) -> Output(10)

Every weight update below is a REAL gradient computed by explicit backprop
math (chain rule), not a framework doing it invisibly. That's the point:
you can print any dW, db, z, or a at any layer and see exactly what it is.

Dataset: sklearn's built-in `digits` dataset (8x8 = 64 pixel images of
handwritten digits 0-9, 1797 samples). This is NOT MNIST (28x28, 70k images)
because this sandbox has no network path to download MNIST. It's the same
task (grayscale handwritten digit classification), just smaller. Swap in
real MNIST via `keras.datasets.mnist.load_data()` on your own machine and
everything else here is unchanged except INPUT_SIZE = 784.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
digits = load_digits()
X = digits.data / 16.0          # pixel values are 0-16; scale to [0,1]
y = digits.target                # integer labels 0-9

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Input dim: {X_train.shape[1]} (8x8 flattened image)\n")

# ---------------------------------------------------------------------------
# 2. ARCHITECTURE
# ---------------------------------------------------------------------------
LAYER_SIZES = [64, 16, 32, 64, 32, 16, 10]   # input, H1..H5, output
NUM_LAYERS = len(LAYER_SIZES) - 1             # number of weight matrices = 6

# ---------------------------------------------------------------------------
# 3. WEIGHT / BIAS INITIALIZATION
# ---------------------------------------------------------------------------
# Weights are NOT calculated by any formula — they start as small random
# numbers and are corrected iteratively by gradient descent. The only
# "formula" involved is the SCALE of the randomness, chosen to keep the
# variance of activations roughly constant across layers so gradients
# neither vanish nor explode early in training.
#
#   He initialization (for ReLU layers): std = sqrt(2 / fan_in)
#   Xavier initialization (for the softmax/output layer): std = sqrt(1 / fan_in)
#
# Biases start at exactly zero — there's no symmetry-breaking problem with
# biases (that problem only applies to weights: if all weights in a layer
# started identical, every neuron in that layer would compute the same
# gradient forever and stay identical — "symmetry"). Zero-initialized
# weights would break training completely; zero-initialized biases are fine.

def init_params(layer_sizes):
    params = {}
    for l in range(1, len(layer_sizes)):
        fan_in = layer_sizes[l - 1]
        fan_out = layer_sizes[l]
        if l < len(layer_sizes) - 1:
            std = np.sqrt(2.0 / fan_in)          # He init, hidden (ReLU) layers
        else:
            std = np.sqrt(1.0 / fan_in)          # Xavier init, output (softmax) layer
        params[f"W{l}"] = np.random.randn(fan_out, fan_in) * std
        params[f"b{l}"] = np.zeros((fan_out, 1))
    return params

params = init_params(LAYER_SIZES)

print("Initial parameter shapes (this is the 'is it randomly assigned' answer):")
for l in range(1, NUM_LAYERS + 1):
    print(f"  W{l}: {params[f'W{l}'].shape}   b{l}: {params[f'b{l}'].shape}")
print()
print("Sample of ACTUAL random initial weights, first hidden neuron (H1, neuron 0):")
print(f"  W1[0, :5] = {params['W1'][0, :5]}   (first 5 of 64 input weights)")
print(f"  b1[0]     = {params['b1'][0, 0]}")
print()

# ---------------------------------------------------------------------------
# 4. ACTIVATION FUNCTIONS
# ---------------------------------------------------------------------------
# Applied ELEMENTWISE. Mathematically the activation is a per-neuron scalar
# function g: R -> R, applied independently to each neuron's own pre-
# activation value z_i. There is no mixing between neurons in this step
# (mixing already happened in the matmul that produced z). Computationally
# we apply it to the whole layer's z-vector at once (vectorized), but that's
# an implementation detail — conceptually and mathematically it is a
# per-neuron operation, always AFTER the weighted sum + bias, never before,
# and never combined across neurons.

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    # z shape: (10, batch_size) - one column per example
    z_shift = z - np.max(z, axis=0, keepdims=True)   # numerical stability
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# ---------------------------------------------------------------------------
# 5. FORWARD PASS
# ---------------------------------------------------------------------------
# For every layer l:
#     z^[l] = W^[l] . a^[l-1] + b^[l]      <- linear combination (the "perceptron" math)
#     a^[l] = g( z^[l] )                    <- activation applied elementwise
#
# a^[0] is just the input X itself.
# For hidden layers (1-5): g = ReLU
# For the output layer (6): g = softmax  (turns 10 raw scores into a
#     probability distribution over the 10 digit classes that sums to 1 —
#     this is why sigmoid would have been wrong here)

def forward(X, params):
    cache = {"a0": X.T}   # transpose so each COLUMN is one example: (features, batch)
    a = X.T
    for l in range(1, NUM_LAYERS + 1):
        W, b = params[f"W{l}"], params[f"b{l}"]
        z = W @ a + b
        if l < NUM_LAYERS:
            a = relu(z)
        else:
            a = softmax(z)
        cache[f"z{l}"] = z
        cache[f"a{l}"] = a
    return a, cache   # a is final output (10, batch) of class probabilities

# ---------------------------------------------------------------------------
# 6. LOSS
# ---------------------------------------------------------------------------
def cross_entropy_loss(Y_pred, Y_true):
    # Y_pred, Y_true shape: (10, batch)
    m = Y_true.shape[1]
    eps = 1e-9
    return -np.sum(Y_true * np.log(Y_pred + eps)) / m

# ---------------------------------------------------------------------------
# 7. BACKWARD PASS  (explicit chain rule, no autograd)
# ---------------------------------------------------------------------------
# Core identity used at every layer:
#     dL/dW^[l] = dL/dz^[l] . (a^[l-1])^T
#     dL/db^[l] = sum over batch of dL/dz^[l]
#     dL/da^[l-1] = (W^[l])^T . dL/dz^[l]      <- this is how error flows BACKWARD
#
# For the output layer with softmax + cross-entropy, the combined gradient
# collapses beautifully to:
#     dL/dz^[output] = a^[output] - y_true       (a genuinely elegant result —
#                                                   worth knowing why: the log
#                                                   in cross-entropy cancels
#                                                   the exp in softmax)
#
# For hidden layers:
#     dL/dz^[l] = dL/da^[l] * g'(z^[l])           <- elementwise multiply by
#                                                    the activation's own
#                                                    derivative (ReLU' here)

def backward(X, Y_true, params, cache):
    m = X.shape[0]
    grads = {}
    dz = cache[f"a{NUM_LAYERS}"] - Y_true.T   # softmax+CE shortcut, shape (10, batch)

    for l in range(NUM_LAYERS, 0, -1):
        a_prev = cache[f"a{l-1}"]
        grads[f"dW{l}"] = (dz @ a_prev.T) / m
        grads[f"db{l}"] = np.sum(dz, axis=1, keepdims=True) / m
        if l > 1:
            da_prev = params[f"W{l}"].T @ dz
            dz = da_prev * relu_grad(cache[f"z{l-1}"])
    return grads

# ---------------------------------------------------------------------------
# 8. PARAMETER UPDATE (plain gradient descent)
# ---------------------------------------------------------------------------
def update_params(params, grads, lr):
    for l in range(1, NUM_LAYERS + 1):
        params[f"W{l}"] -= lr * grads[f"dW{l}"]
        params[f"b{l}"] -= lr * grads[f"db{l}"]
    return params

def accuracy(X, y_true, params):
    Y_pred, _ = forward(X, params)
    preds = np.argmax(Y_pred, axis=0)
    return np.mean(preds == y_true)

# ---------------------------------------------------------------------------
# 9. TRAINING LOOP -- 10 EPOCHS
# ---------------------------------------------------------------------------
# One EPOCH = one full pass through the entire training set.
# One ITERATION (here) = one mini-batch update within that pass.
# So with batch_size=32 and ~1437 training examples, each epoch has
# ceil(1437/32) ≈ 45 iterations. The weights are updated 45 times PER EPOCH,
# not once per epoch -- "epoch" and "iteration" are not the same thing,
# and getting this distinction wrong is a common source of confusion.

BATCH_SIZE = 32
LR = 0.1
EPOCHS = 10

n = X_train.shape[0]
print(f"Training: batch_size={BATCH_SIZE}, lr={LR}, epochs={EPOCHS}")
print(f"Iterations (weight updates) per epoch: {int(np.ceil(n / BATCH_SIZE))}\n")

print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8}")
print("-" * 42)

for epoch in range(1, EPOCHS + 1):
    perm = np.random.permutation(n)
    X_shuf, Y_shuf = X_train[perm], Y_train[perm]

    epoch_loss = 0.0
    num_batches = 0
    for start in range(0, n, BATCH_SIZE):
        end = start + BATCH_SIZE
        X_batch = X_shuf[start:end]
        Y_batch = Y_shuf[start:end]

        Y_pred, cache = forward(X_batch, params)          # forward pass
        loss = cross_entropy_loss(Y_pred, Y_batch.T)       # measure error
        grads = backward(X_batch, Y_batch, params, cache)  # backward pass
        params = update_params(params, grads, LR)          # update weights

        epoch_loss += loss
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    train_acc = accuracy(X_train, y_train, params)
    test_acc = accuracy(X_test, y_test, params)
    print(f"{epoch:>5} | {avg_loss:>10.4f} | {train_acc:>9.4f} | {test_acc:>8.4f}")

print()
print("Weight snapshot AFTER training, same neuron as before (H1, neuron 0):")
print(f"  W1[0, :5] = {params['W1'][0, :5]}")
print(f"  b1[0]     = {params['b1'][0, 0]}")
print("(compare to the initial random values printed above -- this IS learning:")
print(" the same 5 numbers, moved by gradient descent, nothing more mystical than that.)")

# ---------------------------------------------------------------------------
# 10. TRACE ONE SINGLE FORWARD PASS THROUGH ONE NEURON, ONE EXAMPLE, IN FULL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FULL TRACE: one real example through one real neuron, post-training")
print("=" * 70)
x0 = X_test[0:1]              # one test image, shape (1, 64)
true_label = y_test[0]

_, trace_cache = forward(x0, params)

z1_neuron0 = trace_cache["z1"][0, 0]
a1_neuron0 = trace_cache["a1"][0, 0]
print(f"Layer 1, neuron 0 (out of 16):")
print(f"  z = W1[0,:] . x + b1[0] = {z1_neuron0:.4f}   (raw weighted sum + bias)")
print(f"  a = ReLU(z) = max(0, {z1_neuron0:.4f}) = {a1_neuron0:.4f}   (after activation)")

final_probs = trace_cache[f"a{NUM_LAYERS}"][:, 0]
predicted = np.argmax(final_probs)
print(f"\nFinal layer output (softmax probabilities over digits 0-9):")
for digit in range(10):
    marker = "  <-- predicted" if digit == predicted else ""
    print(f"  P(digit={digit}) = {final_probs[digit]:.4f}{marker}")
print(f"\nTrue label: {true_label}  |  Predicted: {predicted}  |  {'CORRECT' if predicted == true_label else 'WRONG'}")