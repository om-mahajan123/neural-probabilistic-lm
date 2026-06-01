import torch
import math
from utils import build_vocab

HIDDEN_LAYER = 700
EMBEDDING_SPACE = 128
CONTEXT_LENGTH = 3

#Intializes all parameters for the model
def init_params(vocab_size):
    a1 = math.sqrt(6 / (EMBEDDING_SPACE * CONTEXT_LENGTH + HIDDEN_LAYER))
    a2 = math.sqrt(6 / (HIDDEN_LAYER + vocab_size))

    C = torch.rand((vocab_size, EMBEDDING_SPACE)) * 0.2 - 0.1
    w1 = torch.rand((EMBEDDING_SPACE * CONTEXT_LENGTH, HIDDEN_LAYER)) * (2 * a1) - a1
    w2 = torch.rand((HIDDEN_LAYER, vocab_size)) * (2 * a2) - a2
    b1 = torch.zeros(HIDDEN_LAYER)
    b2 = torch.zeros(vocab_size)

    return C, w1, b1, w2, b2

#Forward pass of the model
def forward(batch, batch_labels, C, w1, b1, w2, b2):
        z1 = batch @ w1 + b1
        a1 = torch.tanh(z1)          
        
        z2 = a1 @ w2 + b2
        probs = torch.softmax(z2, dim=1) 
        
        loss = None
        if batch_labels is not None:
            correct_probs = probs[torch.arange(batch_labels.shape[0]), batch_labels]
            loss = -torch.log(correct_probs).mean()

        return probs, loss, z1, a1

#This function returns the perplexity/'confusion' of the model
def evaluate(X_val, Y_val, C, w1, b1, w2, b2, batch_size):
    val_batch_count = len(X_val) // batch_size
    total_perplexity = 0

    with torch.no_grad():
        for vb in range(val_batch_count):
            batch_start = vb * batch_size
            batch_end = batch_start + batch_size

            batch_X = X_val[batch_start:batch_end].long()
            batch_Y = Y_val[batch_start:batch_end].long()

            batch_embedded = C[batch_X].view(batch_size, EMBEDDING_SPACE * CONTEXT_LENGTH)
            _, loss, _, _ = forward(batch_embedded, batch_Y, C, w1, b1, w2, b2)
            total_perplexity += torch.exp(loss)

    return (total_perplexity / val_batch_count).item()