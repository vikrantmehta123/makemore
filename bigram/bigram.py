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