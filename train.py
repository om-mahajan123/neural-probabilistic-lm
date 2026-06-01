import torch
import os
import argparse
from data import load_data
from model import init_params, forward, evaluate, EMBEDDING_SPACE, CONTEXT_LENGTH

#This will be the folder the final parameters get stored
os.makedirs('checkpoints', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--new_params', action='store_true', help='Initialize fresh parameters instead of loading from checkpoint')
args = parser.parse_args()

# Hyperparameters
EPOCHS = 1
BATCH_SIZE = 512
LR = 1
LR_FINAL = 0.01

# Load data and initialize parameters
X_train, Y_train, X_val, Y_val, X_test, Y_test, ix_to_word, word_to_ix, vocab_size = load_data('data/text_chunk.txt')
print("Created Train, Validation, Test Sets")

if args.new_params or not os.path.exists('checkpoints/model.pth'):
    print("Initializing new parameters...")
    C, w1, b1, w2, b2 = init_params(vocab_size)
else:
    print("Loading existing parameters...")
    checkpoint = torch.load('checkpoints/model.pth')
    C  = checkpoint['C']
    w1 = checkpoint['w1']
    b1 = checkpoint['b1']
    w2 = checkpoint['w2']
    b2 = checkpoint['b2']

batch_count = len(X_train) // BATCH_SIZE
lr = LR

print("Starting Training:\n")
for epoch in range(EPOCHS):
    # Shuffle each epoch
    perm = torch.randperm(X_train.shape[0])
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    total_loss = 0
    print(f'Epoch: {epoch} | Learning Rate: {lr}')

    for b in range(batch_count):
        batch_start = b * BATCH_SIZE
        batch_end = batch_start + BATCH_SIZE

        batch_X = X_train[batch_start:batch_end]
        batch_Y = Y_train[batch_start:batch_end]

        batch_embedded = C[batch_X].view(BATCH_SIZE, EMBEDDING_SPACE * CONTEXT_LENGTH)

        # Forward pass
        probs, loss, z1, h = forward(batch_embedded, batch_Y, C, w1, b1, w2, b2)
        total_loss += loss.item()

        # Backward pass
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

        if b % 300 == 0:
            print(f"Average CE Loss: {total_loss / (b+1):.3f} | Batch: {b+1}/{batch_count}")

    # Decay learning rate each epoch
    lr = LR - (LR - LR_FINAL) * ((epoch + 1) / EPOCHS)
    val_perplexity = evaluate(X_val, Y_val, C, w1, b1, w2, b2, BATCH_SIZE)
    print(f"Epoch {epoch} Finished | Avg Train Loss: {total_loss/batch_count:.4f} | Val Perplexity: {val_perplexity:.2f}")

#Saving our final parameters in model.pth file
print("Finished Training - Saving all parameters to model.pth")
torch.save({
    'C': C,
    'w1': w1,
    'b1': b1,
    'w2': w2,
    'b2': b2,
    'word_to_ix': word_to_ix,
    'ix_to_word': ix_to_word,
}, 'checkpoints/model.pth')