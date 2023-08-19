import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

# Hyperparameters 
batch_size = 32
block_size = 16
training_iters = 5000
num_generated_tokens = 500
eval_interval = 1000 
learning_rate = 1e-3
device = 'cpu' #'mps' if torch.backends.mps.is_built() else 'cpu'
eval_iters = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.2


# Import text data 
file = open('tinyShakespeare.txt', 'r')
text = file.read()

# Create string to/from integer mapping
characters = ''.join(sorted((list(set(text)))))
vocab_size = len(characters)
stoi = {c:i for i, c in enumerate(characters)}
itos = {i:c for i, c in enumerate(characters)}

# Encoding/Decoding functions to/from integer/character
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[n] for n in l])

# Convert text into a encoded torch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Create training and testing split of data 
train_size = int(len(text) * .9)
train_data = data[:train_size]
test_data = data[train_size:]

# Function that create batches
def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device) 

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size)
            logits, loss = model.forward(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Self attention head
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

# Multi head attention
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Linear then non-linear layer 
class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigram model class 
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, inputs, targets=None):
        B,T = inputs.shape

        token_embd = self.token_embedding_table(inputs)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx 

# Create the BigramLanguageModel and optimizer
model = BigramLanguageModel(vocab_size)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Train model 
for step in tqdm(range(training_iters)):
    # Get a batch to train on
    x, y = get_batch('train', batch_size, block_size)
    
    # Reset the parameter gradients 
    optimizer.zero_grad(set_to_none=True)

    # Forward Pass 
    logits, loss = model.forward(x, y)
    if (step % eval_interval) == 0:
        out = estimate_loss(model)
        tqdm.write(f"At step {step} train loss = {out['train']}, test loss = {out['test']}")

    # Backward Pass 
    loss.backward() 
    optimizer.step()
    

# Generate text from the model
print(f"Generating {num_generated_tokens} tokens of text...\n")
start_idx = torch.zeros((1,1), dtype=torch.long, device=device)
generation = model.generate(idx=start_idx, max_new_tokens=num_generated_tokens)[0]
decoded_generation = decode(generation.tolist())
print(decoded_generation)