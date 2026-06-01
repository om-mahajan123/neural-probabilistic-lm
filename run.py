import torch
import argparse
from model import forward, EMBEDDING_SPACE, CONTEXT_LENGTH

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, required=True, help='All lowercase space separated list of {CONTEXT_LENGTH} words')
args = parser.parse_args()

# Load checkpoint
checkpoint = torch.load('checkpoints/model.pth')
C  = checkpoint['C']
w1 = checkpoint['w1']
b1 = checkpoint['b1']
w2 = checkpoint['w2']
b2 = checkpoint['b2']
word_to_ix = checkpoint['word_to_ix']
ix_to_word = checkpoint['ix_to_word']

# Parse prompt
words = args.prompt.strip().split()
if len(words) != CONTEXT_LENGTH:
    print("Please provide exactly 3 words")
    exit()

# Check all words are in vocab
for w in words:
    if w not in word_to_ix:
        print(f"'{w}' not in vocabulary")
        exit()

# Run inference
indices = torch.tensor([[word_to_ix[w] for w in words]])
embedded = C[indices].view(1, EMBEDDING_SPACE * CONTEXT_LENGTH)

with torch.no_grad():
    probs, _, _, _ = forward(embedded, None, C, w1, b1, w2, b2)

# Get top 5 predictions
top5_probs, top5_ix = torch.topk(probs[0], 5)
print(f"\nInput: '{' '.join(words)}'")
print(f"Top 5 predictions:")
for prob, ix in zip(top5_probs, top5_ix):
    print(f"  {ix_to_word[ix.item()]:<10} {prob.item()*100:.2f}%")