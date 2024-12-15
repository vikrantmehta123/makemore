# imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# Globals
batch_size = 64
block_size = 256
epochs = 5000
eval_interval = 1000
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_blocks = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# create dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create an integer to character mapping- i.e. the tokenizer that encodes and decodes tokens
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]  # takes an input string, and outputs a list of integers. i.e. the character map
decode = lambda l: "".join([itos[i] for i in l]) # takes the token list, and produces the string for it


# Split train-validation
data = torch.tensor(encode(text), dtype=torch.long)
cut = int(0.9 * len(data))
train_data = data[:cut]
validation_data = data[cut:]

# Function to split data into batches
def get_batch(split:str):
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size, )) # randomly select batch_size many indices. len(data) - block_size just handles edge case
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

# model classes
class Head(nn.Module):
    """Single head of self-attention"""

    def __init__(self, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2, -1) * C**-0.5

        # TODO: need to understand indexing in tril for this line. It's most likely to handle less than block_size inputs
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf') )
        weights = F.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        v = self.value(x)

        out = weights @ v
        return out
    
class Block(nn.Module):
    """Single Transformer block"""

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head # make head size smaller based on num of heads
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # branch off, do some computation (LayerNorm applied before transformation) and come back i.e. skip connection
        x = x + self.ffwd(self.ln2(x)) # branch off, do some computation (LayerNorm applied before transformation) and come back i.e. skip connection

        return x

class MultiHeadedAttention(nn.Module):
    """Multiple attention heads running in parallel"""

    def __init__(self, num_heads, head_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # all heads run in parallel
        self.proj = nn.Linear( n_embd, n_embd) # (the output of self-attention, the required output). In this case, both are same. But they need not be
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))

        return out
    
class FeedForward(nn.Module):
    """Simple MLP followed by non-linearity"""

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(), 
            nn.Linear( 4 * n_embd, n_embd), # (dim of output of ffwd, the required dim). In this case, both are same. But they need not be
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class LanguageModelTransformer(nn.Module):

    def __init__(self, n_head, n_blocks, block_size, n_embd):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential( *[Block(n_embd, n_head) for _ in range(n_blocks)]  )

        self.ln_final = nn.LayerNorm(n_embd)

        # Final linear layer to produce logits of equal dim as targets
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Combine the information of the token and the position
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # (B, T, C)

        # pass the position + token information through the attention heads
        x = self.blocks(x) # For this example, the output is: (B, T, 32)

        x = self.ln_final(x)

        # produce logits
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            
            # keep only the last block_size tokens
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond) # logits.shape is (4, x, 65)

            # same as bigram
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)

            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        
        # idx will be the sequence generated for each batch
        return idx
    

model = LanguageModelTransformer(n_blocks=4, n_head=4, block_size=8, n_embd=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# function to get a smoother estimate of the loss
@torch.no_grad()
def estimate_loss():
    out = { }

    # set model to eval mode
    model.eval()

    # for train and val data, take mean of 300 iters
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    # set model back to train mode
    model.train()
    return out

# training loop
for epoch in range(epochs):
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {epoch}: train loss {losses['train']:.4f}, and val loss:{losses['val']:.4f}")
    
    # sample a batch from the dataset
    xb, yb = get_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Loss is: {loss.item():.4f}\n")
print("Generated sequence: ")
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=250)[0].tolist()))
