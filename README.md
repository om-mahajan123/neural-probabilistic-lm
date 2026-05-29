# neural-probabilistic-lm

A from-scratch implementation of the core architecture from Bengio et al. (2003), ["A Neural Probabilistic Language Model"](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Built to actually understand what's going on under the hood, I manually
wrote out each derivtaive to fully understand everything

---

## What this is

The goal of the paper was to train a neural network to predict the next word given the previous *n* words. There is an interesting byproduct that happens though. The model learns **word embeddings** which are vector representations of words where semantically similar words (like *pencil* and *pen*) end up close together in vector space aka embeddings space.

Before embeddings, the standard approach was using one-hot encoding which is a massive vector with a 1 in one position and 0s everywhere else. For a vocab of 10,000 words that's a 10,000-dimensional input, and *shirt* and *pants* are just as "similar" as *pants* and *lamp*. Embeddings solve both problems by using smaller dimenional representations and semantic meaning gets represnted by the learned vectors.

The model here uses the previous 3 words to predict the next one, learns 128-dimensional embeddings, and is trained with a manually derived backprop pass (no `loss.backward()`).

---

## Architecture

```
Input: 3 word indices
  → Embedding lookup (128D each) → concat → 384D vector
  → Hidden layer (700 neurons, tanh)
  → Output layer (vocab_size neurons, softmax)
  → Cross-entropy loss
```

Weights are initialized with Xavier uniform, embeddings with uniform `[-0.1, 0.1]`.

**Where this differs from the paper:**
- The original paper includes a direct connection from the embedding layer to the output layer, I left this out
- The paper uses larger context windows (5-10), we use 3
- Minor differences in initialization and training setup

---

## File structure

```
neural-probabilistic-lm/
├── data/
│   └── text_chunk.txt       # not included, see below
├── notebooks/
│   └── notebook.ipynb       # step-by-step walkthrough with all the math + manual backprop
├── utils.py                 # vocab building, word <-> index mappings
├── data.py                  # loads text, builds X/Y tensors
├── model.py                 # parameter initialization + forward pass
├── train.py                 # training loop with manual backprop
├── requirements.txt
└── README.md
```

The `notebooks/` folder is worth a look if you want to see the full derivation of each gradient, it is a little messy but you can see all the math
behind the backprop as well as the matrix shapes

---

## Data

This project was trained on the **Penn Treebank (PTB)** dataset, a standard famous NLP dataset. It is not included in this repo due to size and licensing.

You can download a preprocessed version here:
- [PTB via Tomáš Mikolov's page](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)

Once downloaded, place it in `data/text_chunk.txt`.

---

## Setup

```bash
git clone https://github.com/yourusername/neural-probabilistic-lm
cd neural-probabilistic-lm

python -m venv venv
source venv/bin/activate       

pip install -r requirements.txt
```

---

## Run

```bash
python train.py
```

You should see loss dropping from around ~9.2 down toward ~5.3 over 20 epochs:

```
Epoch 1/20 | Batch 0/1733 | Loss: 9.1842
Epoch 1/20 | Batch 100/1733 | Loss: 7.6201
...
Epoch 20/20 | Batch 1700/1733 | Loss: 5.3104
```

You can expirement with the learning rate and other hyperparameters to see how they affect with model training

---

## References

Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*, 3, 1137–1155.