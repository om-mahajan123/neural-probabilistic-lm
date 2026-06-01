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
  → Total parameters comes out to around ~7.7M
```

Weights are initialized with Xavier uniform, embeddings with uniform `[-0.1, 0.1]`.

**Where this differs from the paper:**
- The original paper includes a direct connection from the embedding layer to the output layer, I left this out
- The paper uses stochastic gradient ascent on the log-likelihood directly, I minimize CE loss with mini-batch gradient descent and a decaying learning rate
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

## Evaluation

Model confidence is tracked using perplexity, computed on a held-out validation set at the end of each epoch. Perplexity measures how "surprised" the model is by the validation data. A perplexity of 20 means the model is as confused as if it had to pick uniformly from 20 words at every step. Lower is better. The model gets heavily penalized if it assigns low probability to the correct word.
After 20 epochs this implementation reaches a validation perplexity of around ~280-300, which is consistent with what the original paper reports

Also one thing to note while training the model is that except loss to bounce around a little bit. This is the nature of mini-batch GD because not all mini-batches are equal as some may have unique or 'harder' words. This fluctuation can actually be good as it can 'jump' out of a local minima sometimes

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

## Training

By default, `train.py` will load existing parameters from `checkpoints/model.pth` and continue training from where it left off. If no checkpoint exists yet it will create new fresh parameters.

```bash
python train.py              # load existing checkpoint and continue training
python train.py --new_params # creates new parameters regardless
```

Parameters and the embedding matrix is saved to `checkpoints/model.pth` at the end of each run. This means you can stop and resume training at any point without losing progress.

You should see loss and perplexity dropping across epochs:

```
Epoch: 0 | Learning Rate: 0.1
Average CE Loss: 9.210 | Batch: 1/1386
Average CE Loss: 7.467 | Batch: 1201/1386
Epoch 0 Finished | Avg Train Loss: 7.3666 | Val Perplexity: 837.92

Epoch: 5 | Learning Rate: 0.0725
Epoch 5 Finished | Avg Train Loss: 6.1203 | Val Perplexity: 487.33

Epoch: 20 | Learning Rate: 0.01
Epoch 20 Finished | Avg Train Loss: 5.3104 | Val Perplexity: ~280-300
```

---

## Inference

Once training is complete, run inference on any 3-word prompt:

```bash
python run.py --prompt "the dog ran"
```

Output:
```
Input: 'the dog ran'
Top 5 predictions:
  in              5.21%
  the             4.61%
  a               4.18%
  to              4.12%
  <unk>           3.77%
```

All 3 words must be present in the PTB vocabulary. The model will notify you if a word is out of vocabulary.

---

## Future Work

There are a few directions I'd like to extend this project:

- Word2Vec comparison, implement CBOW and skip-gram from scratch and compare the learned embedding spaces against Bengio's model
- Embedding visualization, t-SNE plots of the learned embedding space to visually verify that semantically similar words cluster together
- Analogy evaluation, test the king - man + woman = queen style analogies using cosine similarity on the learned embeddings
- Transformer, my next and current step is the Transformer implementatio now

---

## References

Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*, 3, 1137–1155.
