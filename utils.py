def build_vocab(words):
    unique_words = sorted(set(words))
    
    ix_to_word = {i: word for i, word in enumerate(unique_words)}
    word_to_ix = {word: i for i, word in ix_to_word.items()}
    vocab_size = len(unique_words)
    
    return ix_to_word, word_to_ix, vocab_size