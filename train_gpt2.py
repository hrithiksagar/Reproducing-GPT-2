from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384 
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config 
        
        # wte = weight, token mebeddings
        # pe = posiitonal embeddings 
        # this is the skeleton that represents original transformer architecture with few tweakings in the positions of the Attention heads, linear layers and multi head attensions 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd), # this is the new layer added by GPT 2 only 
            
        ))
        self.lm_head = nn.Layer(config.n_embd, config.vocab_size, bias = False) # Linear layer in the top of the transformer