import torch
from torch import nn
from torch.nn import functional as F

# Hyperparameters 
batch_size = 32
block_size = 8
training_iters = 10000
num_generated_tokens = 500
eval_interval = 300 
learning_rate = 1e-2
device = 'cpu' #'mps' if torch.has_mps else 'cpu'
eval_iters = 200 


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


# Bigram model class 
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs)

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
            logits, loss = self(idx)
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
for step in range(training_iters):
    # Get a batch to train on
    x, y = get_batch('train', batch_size, block_size)
    
    # Reset the parameter gradients 
    optimizer.zero_grad(set_to_none=True)

    # Forward Pass 
    logits, loss = model.forward(x, y)
    if (step % eval_interval) == 0:
        out = estimate_loss(model)
        print(f"At step {step} train loss = {out['train']}, test loss = {out['test']}")

    # Backward Pass 
    loss.backward() 
    optimizer.step()
    

# Generate text from the model
print(f"Generating {num_generated_tokens} tokens of text...\n")
start_idx = torch.zeros((1,1), dtype=torch.long, device=device)
generation = model.generate(idx=start_idx, max_new_tokens=num_generated_tokens)[0]
decoded_generation = decode(generation.tolist())
print(decoded_generation)