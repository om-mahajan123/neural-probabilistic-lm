import torch
import math
from utils import build_vocab

HIDDEN_LAYER = 700
EMBEDDING_SPACE = 128
CONTEXT_LENGTH = 3

def init_params(vocab_size):
    a1 = math.sqrt(6 / (EMBEDDING_SPACE * CONTEXT_LENGTH + HIDDEN_LAYER))
    a2 = math.sqrt(6 / (HIDDEN_LAYER + vocab_size))

    C = torch.rand((vocab_size, EMBEDDING_SPACE)) * 0.2 - 0.1
    w1 = torch.rand((EMBEDDING_SPACE * CONTEXT_LENGTH, HIDDEN_LAYER)) * (2 * a1) - a1
    w2 = torch.rand((HIDDEN_LAYER, vocab_size)) * (2 * a2) - a2
    b1 = torch.zeros(HIDDEN_LAYER)
    b2 = torch.zeros(vocab_size)

    return C, w1, b1, w2, b2

def forward(batch, batch_labels, C, w1, b1, w2, b2):
        z1 = batch @ w1 + b1
        a1 = torch.tanh(z1)          
        
        z2 = a1 @ w2 + b2
        probs = torch.softmax(z2, dim=1) 
        
        #Loss function (Cross-Entropy)
        correct_probs = probs[torch.arange(batch_labels.shape[0]), batch_labels]
        loss = -torch.log(correct_probs).mean()

        return probs, loss, z1, a1