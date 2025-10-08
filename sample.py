import tiktoken
import torch

# make model
import GPT2_LM as X
model = X.GPT2_LM(X.GPTConfig())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f'{sum([p.numel() for p in model.parameters()]) / 1e06:.2f}M parameters')

def generate(num_return_sequences=3, prompt="what is a large language model?"):
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    for i in range(num_return_sequences):
        tokens = model.generate(x)[i, :].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

# train
from train import Train
config = {
    'model': model,
    'max_lr': 6e-4,
    'min_lr': 3e-5,
    'warmup_steps': 10,
    'weight_decay': 0.1,
    'max_steps': 50,
    'batch_size': 2**19,
    'B': 16,
    'T': 1024,
    'filepath': 'input.txt',
}

tr = Train(config)
# tr.train()