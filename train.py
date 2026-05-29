import torch
from data import load_data
from model import init_params, forward, EMBEDDING_SPACE, CONTEXT_LENGTH

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 512
LR = 0.01
LR_FINAL = 0.005

# Load data and initialize parameters
X, Y, ix_to_word, word_to_ix, vocab_size = load_data('data/text_chunk.txt')
C, w1, b1, w2, b2 = init_params(vocab_size)

batch_count = len(X) // BATCH_SIZE
lr = LR

for epoch in range(EPOCHS):
    # Shuffle each epoch
    perm = torch.randperm(X.shape[0])
    X = X[perm]
    Y = Y[perm]

    for b in range(batch_count):
        batch_start = b * BATCH_SIZE
        batch_end   = batch_start + BATCH_SIZE

        batch_X = X[batch_start:batch_end]
        batch_Y = Y[batch_start:batch_end]

        batch_embedded = C[batch_X].view(BATCH_SIZE, EMBEDDING_SPACE * CONTEXT_LENGTH)

        # Forward pass
        probs, loss, z1, h = forward(batch_embedded, batch_Y, C, w1, b1, w2, b2)

        # ---- Backward pass ----
        dz2 = probs.clone()
        dz2[torch.arange(BATCH_SIZE), batch_Y] -= 1
        dz2 /= BATCH_SIZE

        dw2 = h.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ w2.T
        dz1 = da1 * (1 - torch.tanh(z1) ** 2)

        dw1 = batch_embedded.T @ dz1
        db1 = dz1.sum(axis=0)

        dmini_batch = (dz1 @ w1.T).view(BATCH_SIZE, CONTEXT_LENGTH, EMBEDDING_SPACE)
        dC = torch.zeros_like(C)
        dC.index_add_(0, batch_X.view(-1), dmini_batch.view(-1, EMBEDDING_SPACE))

        # Parameter updates
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2
        C  -= lr * dC

        if b % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {b}/{batch_count} | Loss: {loss.item():.4f}")

    # Decay learning rate each epoch
    lr -= (lr - LR_FINAL) * (epoch / EPOCHS)