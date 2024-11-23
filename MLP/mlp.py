import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Fetch data
words = open('names.txt', 'r').read().splitlines()

# Create integer to character, and character to integer mapping
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
# Add <dot> as the special start and end token. Works for this simple dataset.
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Building the Dataset

block_size = 3  # Context Length: how many characters are we taking to predict a new one


def build_dataset(words):
    X, Y = [], []

    for w in words:

        # The integer indexes for the three context letters
        context = [0] * block_size

        for ch in w + ".":
            ix = stoi[ch]

            # Training example: (context, current_character)
            X.append(context)
            Y.append(ix)

            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print("Shapes and dtypes of the dataset are: ")
    # X is a matrix of shape: (len(words), 3), and Y is of shape: len(words)
    print("X: ", X.shape, X.dtype)
    print("Y: ", Y.shape, Y.dtype)
    print("-----")
    return X, Y


X, Y = build_dataset(words=words)

# Vocab size is 27 ( the size of characters. We're not training a word level language model- but a character level)
# We're trying to project a higher dimension into lower dimension: 27 -> 2
C = torch.randn((27, 2))

embedding = C[X]  # embeddings.shape = (X.shape[0], 3, 2)

W1 = torch.randn((6, 100))
b1 = torch.randn(100)

# Matrix multiplication with the embedding. You need .view to fit matmul
pre_act = embedding.view(-1, 6) @  W1 + b1
h = torch.tanh(pre_act)

W2 = torch.randn((100, 27))
b2 = torch.randn(27)

logits = h @ W2 + b2

counts = logits.exp()

probabs = counts / counts.sum(dim=1, keepdim=True)

# Find the probability assigned to the correct class by the model
loss = -probabs[torch.arange(Y.shape[0]), Y].log().mean()

# Recreating the neural network
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

epochs = 10

for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y[ix])
    print(loss.item())
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    for p in parameters:
        p.data -= 0.11 * p.grad

print(loss.item())