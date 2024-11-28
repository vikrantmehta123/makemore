import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# Fetch data
words = open(r'../names.txt', 'r').read().splitlines()

# Create integer to character, and character to integer mapping
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
# Add <dot> as the special start and end token. Works for this simple dataset.
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Pairwise Bigram frequency counting
# N[ix1, ix2] -> How frequently does itos[ix2] appears after itos[ix1] in the dataset
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Plot the bigram frequencies
plt.figure(figsize=(128, 128))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.title("Pairwise Counts")
plt.show()

g = torch.Generator().manual_seed(2147483647)

# Convert each row into a probability distribution

# Add 1 to the counts because some counts may be zero => infinity when taking logs. Thus "smoothing"
# Higher the added number, the more uniform the probability distribution becomes
P = (N + 1).float()

P /= P.sum(dim=1, keepdim=True)

print("Generating names by probability sampling: ")
for i in range(5):
    idx = 0
    out = []
    while True:
        # Sample an index from the probability distribution of the ith row
        p = P[idx]
        idx = torch.multinomial(
            p, num_samples=1, replacement=True, generator=g).item()

        # Keep generating characters till the model samples an end token i.e. a dot
        out.append(itos[idx])
        if idx == 0:
            break
    print("".join(out))

# Evaluating Model Performance and Building a Loss Function

# Think:
# What does the P[i, j] entry mean?
# P[i, j] is the predicted probability that the j'th character appears after i'th character
# For each bigram, P[i, j] represents the "predicted probability"
# For all instances of the training data, you'd want this probab to be close to 1.
# You can use likelihood here-> likelihood is the product of the predicted probabs. You want to maximize this.

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood
# log(a*b*c) = log(a) + log(b) + log(c)

# Computing Loss
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]

        probab = P[ix1, ix2]
        logprobab = torch.log(probab)

        log_likelihood += logprobab  # You add because you've taken log
        n += 1

nll = -log_likelihood
print(f"Average Negative Log Likelihood using count method is:{nll/n}")

# Creating the training dataset for neural network
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]

        # For each bigram, the first character is the 'x' and the second is it's 'label'
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

# Build the Neural Network

# One forward and backward pass
x_encoded = F.one_hot(xs, num_classes=27).float()

W = torch.randn((27, 27), generator=g, requires_grad=True)

logits = (x_encoded @ W)
counts = logits.exp()
probabs = counts / counts.sum(dim=1, keepdim=True)

loss = -probabs[torch.arange(xs.shape[0]),  ys].log().mean()

print(f"Loss for one sample using neural nets is: {loss.item()}")
W.grad = None
loss.backward()

W.data -= 0.1 * W.grad

# Training the Neural Network
epochs = 100
num = xs.nelement()

for k in range(epochs):

    # forward pass
    # input to the network: one-hot encoding
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts, equivalent to N
    # probabilities for next character
    probs = counts / counts.sum(1, keepdims=True)

    # Add the 0.01 * (W**2).mean() as a regularizer
    # It will force the model to choose W_i such that they are closer to zero
    # It's the equivalent of increasing the "added number" for smoothing in count based approach
    # Closer the Ws to zero, the more uniform distribution you get.
    # 0.01 is the "strength" of regularizer -> equivalent to the "number" you were adding in count-based smoothing
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

    # backward pass
    W.grad = None  # set to zero the gradient
    loss.backward()

    # update
    W.data += -50 * W.grad

print(f"Loss after training the neural net is: {loss.item()}")

# finally, sample from the 'neural net' model
print("Sampling from the neural net: ")

g = torch.Generator().manual_seed(2147483647)

for i in range(5):

    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W  # predict log-counts
        counts = logits.exp()  # counts, equivalent to N
        # probabilities for next character
        p = counts / counts.sum(1, keepdims=True)

        # Sample an index from this "learned" distribution
        ix = torch.multinomial(
            p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
