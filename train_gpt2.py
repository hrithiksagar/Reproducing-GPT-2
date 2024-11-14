from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 
import math

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
    
    def forward(self, x):
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
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # auto regressive mask that makes sure that, the tokens only attend to the tokens before them and NEVER to tokens in the future 
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
        self. c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # 2 linar projections sandwiched between GELU activation, 
        # Dead RELU neuron problem = if any neuron falls in the part where if its flat at 0 all the actions fall at that part there is no development for those activations as its multiplied by 0, this is were gelu changes it by a small curve 
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

#### 2. The below part is written second [BLOCK CLASS]
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # this is still not written, it will be written after MLP (3rd class) as a 4th class
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # this block needs to be implimented still --> 3rd Block
        # happens wiht every single token individualiy, noinfo is bing collected or exchanged between tokens, this is a MAP function
    
    def forward(self, x):
        x = x+ self.attn(self.ln_1(x)) # pre normalization version, layer normalization 1
        x = x+self.attn(self.ln_2(x)) # attentionis a REDUCE function
        return x
    # Can think attention as comunication operation, aggregation/pooling/weighted sum/ reduce operation, all the tokens (1024 tkens lined up) where they exchance the information
    # Both MLP and Attention --> hence called as MAP-REDUCE function

##### 1. The below part is written First [MAIN CONFIGURATION OF TRANSFORMER SKELETON]
@dataclass
class GPTConfig:
    # Commit here: Updated few values from initial commits. 
    block_size: int = 1024  # Max Sequence Length
    vocab_size: int = 50257 # Number of tokens = 50,000 BPE Merges + 256 bytes + 1 <|endoftext|> token
    n_layer: int = 12       # Number of layers
    n_head: int = 7         # Number of heads
    n_embd: int = 768       # Number of dimensions 
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config 
        
        # This is the skeleton that represents original transformer architecture with few tweakings in the positions of the Attention heads, linear layers and multi head attensions 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # wte = weight, token mebeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # wpe = posiitonal embeddings 
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd), # this is the new layer added by GPT 2 only 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False) # Linear layer in the top of the transformer
     
    ##### 6. Main Forward Function is written in this stage [MAIN FORWARD FUNCTION]
    def forward(self, idx, targets=None):
        # idx is of shape (B, T) - indices = idx = our tokens (inputs) = always of shape (B,T) B=Batch Dimension T=Time Dimension and T<B always as B is the minimum sequence length so these are arranged like a 2D layout, every single row is of size B 
        B, T = idx.size()
        assert T <= self.config.block_size, f"cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # Now lets Forward the token and position embeddings:    
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Shape (T) arange = a version of ramge but for pytorch. 
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T,n_embd)
        tok_emb = self.transformer.wte(idx) # toklen embheddingfs of shape (B,T,n_embd)
        x = tok_emb + pos_emb
        
        # Forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # Forward the final LayerNorm and the Classifier
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x)        # Shape: (B,T,Vocal_size) Vocab_Size here is the number of possible tokens, a tensor that we are going to obtain 
        loss = None
        
        if targets is None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1))
        
        return logits, loss
    ##### 6 END of Function and few changes in Main code for testing the model
    
    ##### 5. The below code Classmethod is being added as 5th block [@CLASSMETHOD from_pretrained]
        # from_pretrained is a constructor or class method in python, that retunrs the GPT object of we give GPT type (gpt-2 in our case)
    
    @classmethod
    def from_pretrained(cls, model_type):
        """ 
        Loads pretrained GPT 2 model weights from hugging face
        """    
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained GPT: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
            # these are the hyperparamters of the GPT 2 model
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # this is always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # this is always 1024 for GPT model checkpoints
        
        # create a from scratch initialized miniGPT model
        config = GPTConfig(**config_args) #  GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, block_size=1024), By using **config_args, all these values are passed in a compact way, rather than listing each argument explicitly.
        model = GPT(config)
        sd = model.state_dict() # we are creating state_dict for both our model and hugging face model
        sd_keys = sd.keys() # going over hugging face model keys printed in the second cell of this file "play.ipynb" and copying those over while ignoring few buffers.
        sd_keys = [k for k in sd_keys if not k.endswith('.attn_bias')] # discard thids mask/buffer, not a parameter
        
        # init a huggingface/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters as aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn,weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        #  Also some weights are from tensorflow as originally GPT 2 code is written in tensorflow, so to copy those, they need to match the dimensions which were transposrd, so transposing them back while copying 
        for k in sd_keys_hf:
            print(f"Loading key: {k} | Source shape: {sd_hf[k].shape} | Target shape: {sd[k].shape}")
            
            if k.endswith('attn.c_attn.weight'):
                # Special case for `c_attn` weight in Hugging Face model to match target model's shape
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape[::-1]} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())  # Transpose for compatibility
            elif any(k.endswith(w) for w in transposed):
                # Special treatment for the Conv1D weights that need to be transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Vanilla copy for other parameters
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad(): 
                    sd[k].copy_(sd_hf[k])
                    
            

        return model
    
##----------------------------------------------------------
# model = GPT.from_pretrained('gpt2')
# print("Didn't crash yay!")

## if we did not crash and all the values are exaclty as equal to the original GPT wandbs then we get a confidence that it is working and we can further build the generation code 
# now we should write main forward function: 6

##### 6 Continuation - changed few main running code blocks
num_return_sequences = 5
max_length = 30
model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda') # moving the whole model to GPU from CPU
print("Didn't crash yet!")


##### 7. PREFIX Tokens Creation started here below. 
# so the plan is to make GPT 2 generate a text for a prefix text, that is 
"""Hello, I'm a language model,...""" 

import tiktoken # Library from OpenAI
enc = tiktoken.get_encoding('gpt2') # encoding that is developed for GPT2
tokens = enc.encode("Hello, I'm a language model,")
"""
    The above line will Encode this text to get a list of integer tokens which will look like this: 
    15496, 11, 314, 1101, 257, 3303, 2746, 11   # [Shape = (8,)]
    - got these from this app: https://tiktokenizer.vercel.app/?model=gpt2 for "Hello, I'm a language model,"
"""
tokens = torch.tensor(tokens, dtype = torch.long) # (8, ), those texts that were encoded and converted to list will be tensorized here of 8 tokens hence shape (8, )
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) # (5,8) # now we are repilicating those 8 tokens in the previous step for 5 times, to get 5 rows of 8 tokens which will be the initial input x that will live on GPU 
x = tokens.to('cuda')

# Generate rightg now, x's shape is (B, T), where B=5, T=8
# set the seed to  42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)<max_length:
    # Forward the model to get the logits 
    with torch.no_grad():
        logits = model(x) # (B,T,Vocab_size)
            # Take the logits at the last position
        logits = logits[:,-1,:] # (B, vocab_size)
            # Get the probabilities
        probs = F.softmax(logits, dim=-1)
            # Do top K sampling of 50 (As hugging face kept this as default)
            # all the probabilities after top 50 will be equated to 0
            # topK_probs here becomes (5,50), topK_indinces is (5,50)
        topK_probs, topK_indinces = torch.topk(probs, 50, dim=-1)
            # Select a token from the top-k probabilities
        ix = torch.multinomial(topK_probs, 1) # (B, 1)
            # Gather the corresponding indices
        xcol = torch.gather(topK_indinces, -1, ix) # (B,1)
            # append to the sequence
        x = torch.cat((x,xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded) 
        

