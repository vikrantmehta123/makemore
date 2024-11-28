import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Fetch data
words = open(r'../names.txt', 'r').read().splitlines()

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

# Optimizing the Learning Rate
# Step 1: First, find out the two learning rates by trial & error over which you'll do the following steps. 
#         The first learning rate is where the loss just goes down too rapidly. It's too fast
#         The second learning rate is where the speed of the loss decrease is slightly slower but not too slow.
#         These two are now upper and lower bounds for your learning rates.
# Step 2:

lower, upper = 0.001, 1
learning_rate_exponent = torch.linspace(-3, 0, 1000) # 10^-3 = 0.001
lrs = 10 ** learning_rate_exponent # Candidate learning rates
epochs = 1000

lri = []
lossi = [ ]
for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y[ix])
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    for p in parameters:
        p.data -= lrs[i] * p.grad

    lri.append(learning_rate_exponent[i])
    lossi.append(loss.item())

print(loss.item())


plt.plot(lri, lossi)
plt.show()

# Good learning rate comes out ot be 10*-1 = 0.1
# With this LR, you can iterate for more iterations because now you have confidence that you won't overstep
# nor will you take a very small step
epochs = 10000

for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y[ix])
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad
print(loss.item())

# Near the end of the training, you may want to try slower learning rates -> "Learning Rate Decay"
epochs = 1000

for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y[ix])
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.01
    for p in parameters:
        p.data -= lr * p.grad

print(loss.item())

# Splitting the data into train, validation, and test splits and repeating training
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

X_train, Y_train = build_dataset(words[:n1])
X_val, Y_val = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])

# Reinitialize the Parameters
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# Train neural net on training dataset
epochs = 10000

for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X_train.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X_train[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y_train[ix])
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad

print(loss.item())

# Evaluate on validation set

embedding = C[X_val]
h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
logits = h @ W2 + b2

loss = F.cross_entropy(logits, Y_val)
print("Validation Loss: ", loss.item())


# The training and validation losses are equal => we can make our network more complex without overfitting
# Longer neural net requires to train for longer
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 200), generator=g) # Now, we have 200 neurons in the hidden layer
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

# Train neural net on training dataset
epochs = 10000

for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X_train.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X_train[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 6) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y_train[ix])
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad

print(f"Complex Neural Net Loss: ", loss.item())

# Visualize Embeddings
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

# Increasing Size of Embeddings
C = torch.randn((27, 10))
W1 = torch.randn((30, 200))
b1 = torch.randn(200)
W2 = torch.randn((200, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True


epochs = 10000

for i in range(epochs):
    # Construct mini batch 
    ix = torch.randint(0, X_train.shape[0], (32, )) # Batch of 32

    # Forward pass
    embedding = C[X_train[ix]]

    # Matrix multiplication with the embedding. You need .view to fit matmul
    h = torch.tanh(embedding.view(-1, 30) @  W1 + b1)
    logits = h @ W2 + b2

    loss = F.cross_entropy(logits, Y_train[ix])
    # Backward pass
    for p in parameters:
        p.grad = None
    
    loss.backward()
    lr = 0.1
    for p in parameters:
        p.data -= lr * p.grad

print(f"Higher Embeddings Loss: ", loss.item())

embedding = C[X_val]
h = torch.tanh(embedding.view(-1, 30) @  W1 + b1)
logits = h @ W2 + b2

loss = F.cross_entropy(logits, Y_val)
print("Validation Loss: ", loss.item())

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))