from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 

#### 4. The below ATTENTION Class is added fourth [CAUSAL SELF ATTENTION BLOCK]
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # the below codes will be run only if the assert is True and condition fullfilled.
        # key, query, Value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def foward(self, x):
        """
        Attention operation
        B = Batch sizde
        T = Sequence Length
        n_embd = Embedding dimentionality
        
        Calculate query, key and value for all heads in batch and move head forward to be the batch dim
        nh = number of heads
        hs = head size
        C = number of channels = nh * ns
            For e.g. in GPT-2 [124 M], 
                n_heads = 12
                hs = 64
                C = nh*ns = 12*64 = 768 --> Channels in the transformer
                
        -   instead of having multiple seperate modules that can concatenate all of that is just put into a single self attention module by careflly doing transpose and other simple math functions to make sure there wont be errors. 
            each token emits 3 vectores = Query, key , Value vector
        -   first Q and K must multiple with each other -> to get to know how interesting they find to each other. 
        -   
        """
        B, T, C = x.size()
        qkv = self. c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs) Making number of heads in to a batch dimention, so that in the next operations taht follow, pytorch treats the B nh as batches and applies all tghe operations in parallel in batches and heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # these are the calculations that follow
        att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # auto regressive mask that makes sure that, the tokens only attend to the tokens before them and NEVER to tokens in the future 
        att = F.softmax(att, dim =1 ) # normalizes the attentions so they sums to 1 always
        y = att @ v # attention matrix multiple with values to do weighted sum of values of tokens that were found as interesting the previous step
                    # (B,nh,T,T) * (B,nh,T,hs) --> (B,nh,T,hs)


        # y = F.scaled_dot_product_attention(q,k,v is_casual=True) # FLASH ATTENTION
        y = y.transpose(1,2).contiguous().view(B,T,C) # re-assemble all heads outputs side by side (concatenation operation)
        y = self.c_proj(y)
        return y
                
    
#### 3. The below MLP block is added Third [MLP CLASS]
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4 * config.n_embd )
        self.gelu = nn.GELU(approximate='tanh') # Activation Function, basicallu like RELU but there is not flat tail, smoother version of relu, these days we can use direct version instead of approximation. Just cause GPT 2 used approximation we are using
        self. c_proj = nn.Linear(4 * config)
        # 2 linar projections sandwiched between GELU activation, 
        # Dead RELU neuron problem = if any neuron falls in the part where if its flat at 0 all the actions fall at that part there is no development for those activations as its multiplied by 0, this is were gelu changes it by a small curve 
        
    def foward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#### 2. The below part is written second [BLOCK CLASS]
class Block(nn.Module):
    def __init__(self.config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # this is still not written, it will be written after MLP (3rd class) as a 4th class
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # this block needs to be implimented still --> 3rd Block
        # happens wiht every single token individualiy, noinfo is bing collected or exchanged between tokens, this is a MAP function
    
    def foward(self, x):
        x = x+ self.attn(self.ln_1(x)) # pre normalization version, layer normalization 1
        x = x+self.attn(self.ln_2(x)) # attentionis a REDUCE function
        return x
    # Can think attention as comunication operation, aggregation/pooling/weighted sum/ reduce operation, all the tokens (1024 tkens lined up) where they exchance the information
    # Both MLP and Attention --> hence called as MAP-REDUCE function

##### 1. The below part is written First [MAIN CONFIGURATION OF TRANSFORMER SKELETON]
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
        # This is the skeleton that represents original transformer architecture with few tweakings in the positions of the Attention heads, linear layers and multi head attensions 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd), # this is the new layer added by GPT 2 only 
            
        ))
        self.lm_head = nn.Layer(config.n_embd, config.vocab_size, bias = False) # Linear layer in the top of the transformer