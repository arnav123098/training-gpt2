import math
import numpy as np
from utils import dotdict
import time
import tiktoken
import torch
from torch import nn
import torch.nn.functional as F

# loading data as tokens
class DataLoader:
    def __init__(self, B, T, filepath):
        self.B = B
        self.T = T

        with open(filepath, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        self.current_position = 0

    def next_batch(self):
        if self.current_position + self.B*self.T + 1 > len(self.tokens):
            self.current_position = 0
            
        buf = self.tokens[self.current_position : self.current_position+self.B*self.T+1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        self.current_position += self.B*self.T

        return x, y

class Train:
    def __init__(self, config):
        config = dotdict(config)
        self.max_lr = config.max_lr
        self.min_lr = config.max_lr * config.weight_decay
        self.warmup_steps = config.warmup_steps
        self.max_steps = config.max_steps
        self.model = config.model

        # detecting device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

        torch.manual_seed(1337)
        if device == "cuda":
            torch.cuda.manual_seed(1337)

        # gradient accumulation to simulate large batch size
        self.grad_accum_steps = config.batch_size // (config.B*config.T) if config.batch_size else 1
        print(f'gradient accumulation steps: {self.grad_accum_steps}')

        torch.set_float32_matmul_precision('high')

        self.device = device
        self.optimizer = config.model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=self.max_lr, device=device)
        self.train_loader = DataLoader(config.B, config.T, config.filepath)

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_steps:
            return self.max_lr * (it+1) / self.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.max_steps:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
    def train(self):
        for step in range(self.max_steps):
            t0 = time.time()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss /= self.grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()
            torch.cuda.synchronize() # wait for the GPU to finish work
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps
            tokens_per_sec = tokens_processed / dt
            print(f"step {step:4d} | loss: {loss_accum:.6f} | lr {lr:.4f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
