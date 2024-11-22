import torch
import matplotlib.pyplot as plt

# Fetch data
words = open('names.txt', 'r').read().splitlines()

# Create integer to character, and character to integer mapping
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0   # Add <dot> as the special start and end token. Works for this simple dataset.
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

# # Plot the bigram frequencies
# plt.figure(figsize=(128,128))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#         plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()

g = torch.Generator().manual_seed(2147483647)

# Convert each row into a probability distribution
P = (N).float()
P /= P.sum(dim=1, keepdim=True)

print("Generating names by probability sampling: ")
for i in range(5):
    idx = 0
    out = []
    while True:
        # Sample an index from the probability distribution of the ith row
        p = P[idx] 
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

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

        log_likelihood += logprobab # You add because you've taken log
        n += 1

nll = -log_likelihood
print(f"Average Negative Log Likelihood is:{nll/n}")

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
import torch.nn.functional as F

x_encoded = F.one_hot(xs, num_classes=27).float()

W = torch.randn((27, 27), generator=g)

logits = (x_encoded @ W )
counts = logits.exp()
probabs = counts / counts.sum(dim=1, keepdim=True)

