from utils import build_vocab
import torch

def load_data(path):
    raw_text = open(path, 'r').read()
    words = raw_text.split()
    ix_to_word, word_to_ix, vocab_size = build_vocab(words)

    X = torch.zeros((len(words) - 3, 3), dtype=torch.long)
    Y = torch.zeros(len(words) - 3, dtype=torch.long)

    for i in range(3, len(words)):
        training_instance = torch.tensor([word_to_ix[words[i-3]], word_to_ix[words[i-2]], word_to_ix[words[i-1]]])
        X[i-3] = training_instance
        Y[i-3] = word_to_ix[words[i]] 
    
    train_len = len(X) // 10 * 8
    val_len = len(X) // 10

    X_train = X[:train_len]
    Y_train = Y[:train_len]

    X_val = X[train_len:train_len + val_len]
    Y_val = Y[train_len:train_len + val_len]

    X_test = X[train_len + val_len:]
    Y_test = Y[train_len + val_len:]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, ix_to_word, word_to_ix, vocab_size