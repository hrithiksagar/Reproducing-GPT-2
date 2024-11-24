# Repo link: https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py 
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from torch.nn import functional as F 
import math
import inspect
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from hellaswag import render_example, iterate_examples
import os
os.chdir("/data/circulars/iiith/hrithik/Reproducing-GPT-2/edu_fineweb10B")

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
        self.c_proj.NANOGPT_SCALE_INIT = 1 # Its kind of a flag for this module, attahcing this flag to make sure it wont conflictb with anything
        
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
        
        ##------------Start------
        # these are the calculations that follow
        # att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # auto regressive mask that makes sure that, the tokens only attend to the tokens before them and NEVER to tokens in the future 
        # att = F.softmax(att, dim =1 ) # normalizes the attentions so they sums to 1 always
        # y = att @ v # attention matrix multiple with values to do weighted sum of values of tokens that were found as interesting the previous step
        #             # (B,nh,T,T) * (B,nh,T,hs) --> (B,nh,T,hs)
        ##------------end------
        
        ## Flash Attention 
            #  will be introduced here, it takes these 4 lines and impliments them in a very very fast manner. It is a kernal fusion operation  --> this kernal fusion is the one that "torch.compile" cannot find and fusion it. As it requirs algorithmeic re-write of how attention is imlimented using those 4 lines to merge them and write in lesser lines. Fa does more FLOPS than the attention done in thosen 4 lines here but FA is significantly faster (7.6x faster) Because, FA is very mindful about the memory hierarchy (The GPU element wise calculation, reading and writing to and fro from GPU and CPU stuff discussed in DataLoaderLite Optimizaiton Part, see that for reference) Like whats in HBA whats in SM fusions/orchestrates the computaion in such a way that we have very less reads/writes as in device computations are very very very faster than moving between GPU and CPU.  So even if we are doing lots of FLOPS it is still faster than loading the data between GPU and CPU 
            ## This has reduced the compile time to 50% faster - earlier it was 550ms -  now it is 220ms. 
            # Till now with all the GPU operations such as Torch.compile, flash attention, Moving the logits to bfloat16 from float32 we managed to reduce the time from 1200ms to 220ms. thats very big difference. 

        y = F.scaled_dot_product_attention(q,k,v) #, is_casual=True) # FLASH ATTENTION
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
        self.attn = CausalSelfAttention(config) # this is still not written at that commit (NOW written), it will be written after MLP (3rd class) as a 4th class
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
    n_head: int = 12 # 7         # Number of heads
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
        
        # Weights Sharing Scheme - As per the GPT 2 paper, they have implimented it in this way:
        self.transformer.wte.weight = self.lm_head.weight
            # Here, we are taking the wte.weight and redirectign it to point to the LM_Head, So this copies the data pointer then we will be left with single tensor which is going to be used Twice in the forward pass
            
        # initialize parameters
        self.apply(self._init_weights) # Apply will iterate all the modules of the sub modules and applies _init_weights function on them. 
    
    ### New function - we
    ### Weights initialization: OpenAI GPT 2,3 papers havent mentioned much informaiton about it so have to go thorugh the Official tensorflow based code and have to read in between the lines. 
        # They have used standard deviation of 0.02
        # Initialized 
            # bias with 0
            # Token embeddings with 0.02
            # position embeddings with 0.01 
        # We shall mirror those here
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # Square root of number of residual layers 
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)
        
    
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
        if targets is not None:
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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        # if master_process:
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
##---------------------------------------------------------------------------------------------------------------------------------------
# model = GPT.from_pretrained('gpt2')
# print("Didn't crash yay!")
## if we did not crash and all the values are exaclty as equal to the original GPT wandbs then we get a confidence that it is working and we can further build the generation code 
# now we should write main forward function: 6
##-------------------------------------------------------------------------------------------
import time

# ##### 8. Training loop
# # Attempt to autoconnect to device thats available:
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # APPLE silicon chip
    
# # device = 'cpu' # OVERRIDE

# import tiktoken # Library from OpenAI
# enc = tiktoken.get_encoding('gpt2') # encoding that is developed for GPT2
# with open('input.txt', 'r') as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[:B*T + 1])
# # we cant move buf to device directly, we have to do this:
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)
# x, y = x.to(device), y.to(device)

# # get logits
# model = GPT(GPTConfig())
# model.to(device)
# # logits, loss = model(x,y)

# print("here")
# # print(logits.shape)
# # print(loss)
# optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4) # this learning rate is a very good default learning rate, that we want to run at. avery early debugging stage. 
# for i in range(1000):
#     optimizer.zero_grad() # Always start with zero gradients
#     logits, loss = model(x,y)
#     loss.backward() # Adds to gradients (Gradients + loss) accumilates gradients from the loss 
#     optimizer.step() # step funtion to update the parameters to decrease the loss
#     print(f"step {i}, loss: {loss.item()}") # .item is used becuase loss is a tensor with single element - this will convert it to a single float and float will not live on CPU 
#     # pytorch will the take the single dimension tensors ships it to CPU and convert it to a  float that we can print. 
#     import sys; sys.exit(0)
# """
# this prints:
# using device: cuda
# tensor (11.0886, grad_fn=<NllLossBackward0>) # printed loss
# """
# ####### 8 END -------------------------------------------------------

## -------------------------------------------------------------------

###### 9 Building DATALODER LITE
# the above 8th Block will optimize for 1 batch
# but we want to optimize for XY batches and create a small data loader that makes sure that we are always getting a fresh batch 

# import tiktoken

# class DataLoaderLite:
#     def __init__(self, B, T):
#         self.B = B
#         self.T = T
        
#         # at init load tokens form dick and store them in memory
#         with open('input.txt', 'r') as f:
#             text = f.read()
            
#         # print(f"text type: {type(text)}, text value: {text}")
        
#         enc = tiktoken.get_encoding('gpt2')
#         tokens = enc.encode(text)
#         self.tokens = torch.tensor(tokens)
#         print(f"loaded {len(self.tokens)} tokens")
#         print(f"1 epoch = {len(self.tokens)//(B*T)} batches") # prints number of batches in a single epoch iterating over this dataset  
        
#         # State
#         self.current_position = 0 # start at position 0
            
#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position: self.current_position+B*T+1]
#         x = (buf [:-1]). view(B, T) # inputs
#         y = (buf [1:]).view(B,T) # targets
#         # advance the position in the tensor
#         self. current_position += B * T # its important to always advance our position by B*T 
#         # if loading the next batch would be out of bounds, reset
#         if self.current_position + (B * T + 1) > len(self. tokens): # if we run out of data then we loob back to 0 
#             self.current_position = 0
#         return x, y


# #####------------- Final modified DataLoaderLite for training on real world data-----------------
import numpy as np
def load_tokens(filename):
    print("Attempting to load file 1:", filename)
    npt = np.load(filename)
    # print("Attempting to load file 2:", filename)
    npt = npt.astype(np.int32) # Added after video ended 
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'} #  from the list of files in data_root available, we ony leed train and val 
        
        # get the shard filenames
        # data_root = "edu_fineweb10B"
        data_root = "/data/circulars/iiith/hrithik/Reproducing-GPT-2/edu_fineweb10B"
        shards = os.listdir(data_root) # shards are numpy files, just storing a numpy array similar to torch tensors
        # print("os.listdir(data_root) :",shards) # os.listdir(data_root): ['train_1.npy', 'train_2.npy', 'val_1.npy', 'val_2.npy', 'test_1.npy'] ## prints all the files available 
        shards = [s for s in shards if split in s]
        # # print("shards = [s for s in shards if split in s] :",shards) # As from the list of files available, we ony leed train and val, we are splitting to only find train and val and use them  ; from the list of files available, we ony leed train and val => shards = ['train_1.npy', 'train_2.npy'] and ; shards = ['train_1.npy', 'train_2.npy']=> shards = ['val_1.npy', 'val_2.npy']
        shards = sorted(shards)
        # # print("shards = sorted(shards) :",shards) # sorts the files in alphabetical order, shards = ['train_1.npy', 'train_2.npy']
        
        shads= [os.path.join(data_root, s) for s in shards]
        # print("shads= [os.path.join(data_root, s) for s in shards] :",shards) # shards = ['train_1.npy', 'train_2.npy']; shards = ['edu_fineweb10B/train_1.npy', 'edu_fineweb10B/train_2.npy']
        self.shards = shards # contains full paths of all files: shards = ['train_1.npy', 'train_2.npy']; shards = ['edu_fineweb10B/train_1.npy', 'edu_fineweb10B/train_2.npy']
    
        assert len(shards) > 0, f"No shards found for the split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        
    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
        
# ####-----------------------------------------

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

####---------------------------
# # DISTRIBUTED DATA PARALLEL
#  torchrun --standalone --nproc_per_node=2 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group

# Setup DDP (DISTRIBUTED DATA PARALLEL)
# Torch run commandsets the env variables RANk, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # IS this a DDP run? 
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank. 
    assert torch.cuda.is_available(), " for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_rank = int(os.environ['WORLD_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing
    print("using ddp")
else:
    # vanilla, non-ddp run
    ddp_rank = 0
    ddp_local_rank = 0 
    ddp_world_size = 1
    master_process = True
    # Attempt to autodetect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # APPLE silicon chip
    print(f"Using device: {device}")
    print("Not using ddp")

device_type = 'cuda' if 'cuda' in device else 'cpu'
    
####---------------------------
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# train_loader = DataLoaderLite(B=16, T=1024) # DataLoaderLite(B=4, T=32) changing from this to new higher value sbecause thats what the real world data be like. T =. max_sequence_length of the actual GPT 2 model 
####---------Gradient Accumilation Start----------
## Gradient Accumilation
# GPT 3 used a batch size of 0.5Million, but we cant use tat amount batch size in B = 488 (0.5e6/1024 = 488), it will explode the GPUs, so to fit in such a amount of data we use something called as gradient Accumilations. 

total_batch_size = 524288 # 2**19, ~0.5M in number of tokens
B = 16 # Micro batch Size
T = 1024 # equence Length
assert total_batch_size % (B*T*ddp_world_size) == 0, "make sure total_batch_size is divisible by (B * T * ddp_world_size)"
grad_accum_steps = total_batch_size // (B*T*ddp_world_size) # (2**19)/(16*1024) = 32 
if master_process: 
    print(f"Total desired Batch size: {total_batch_size}")
    print(f"=> Calculated gradient accumilation steps: {grad_accum_steps}")

# print("I am GPU", ddp_rank)
# print("Bye Bye Bye")
# import sys; sys.exit(0)
train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes =ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
# we want multiple GPUS to work on different partsof the data insetad of the all GPUS working on same data 
####---------Gradient Accumilation End----------


# Get logits 
model = GPT(GPTConfig(vocab_size=50304)) # overriding vocab size because - 50257 is a ugly number [Odd number, not much power of 2s] 50304 is a good number with many powers of 2. this is like adding fake tokens in the vocab size. 
    # So just by making the voca size number from Ugly to good we reduced the computation speed by 20ms from 220ms to 199ms 
model.to(device)
# model = torch.compile(model)    ## torch.compile -> basically a compiler for Neural Networks such as GCC for C/C++ code. Extremely simple to use - model = torch.compile(model) . Makes the code lot faster
# # It was 899ms after few speeding up functions but then using torch.compile the time reduced to 550ms thats a very huge differece 
# ## Speed up comes form reducing python overheads and GPU read/writes depending on model arch and batch size 
# # reduce the round trips to/fro from ()GPU to CPU and reciprocal) the memory and does kernal Fusion. First goes through the whole code and then intelligently decides on operations to perform

# enabling TF32 using single line in pytorch
torch.set_float32_matmul_precision('high') # tell pytroch what kind of kernals it should run for matmul 
    # this has decreased the time of optimizing, can be observed if you comment this line and run the optimizwer loop, time taken will be 1200ms, but this will make it to 800ms - so how? tf32 in principle offers a lot faster throughput https://youtu.be/l8pRSuU81PU?t=5966 (go back 20-30 sec) 
    # tf32 will crop out mantissa of the Float32 hence, reducing the size while still being the, look at this image, path: "tf32 bf16.png" 
    
## Learning Rate Scheduler - Cosine Decary Learning Scheduler
    # Taken from GPT 3 papers. Uses Cosine Decary Learning Scheduler with warmup
    # lr starts right at 0 and then linearly ranks up over some amount of time and then comes down with the cosine sort of shape refer the screen shot "lr_cosine_decay.png" come down to minimum learning rate
    
#### MOVING MODEL TO DDP (parallel program) ---- Added almost in the end (3:17 min) 
use_compile = False
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model 

max_lr = 6e-4 # 4e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 10
max_steps = 19073 # 50 # 19,073 steps is ~1 epoch, if data is 10B tokens and
def get_lr(it):
    # 1) linear warm u for warmup_iter steps 
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    # 2) if ir> lr_decay_iters, return min learning rate 
    if it > max_steps:
        return min_lr
    
    # 3) in. between, use cosine decay down to min learning rate 
    decay_ratio = (it - warmup_steps) / (max_lr - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5*(1+math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

    
# Optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps = 1e-8) # added more optimizer parameters, by taking reference from GPT 3 paper as these are not mentioned in GPT 2 papers but We believe that GPT 3 architecture is very similar to GPT2 but have huge dataset. 

# optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate=6e-4, device=device) # this is a new function to be written in class GPT
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device = device) # added almost in the end (3 hr 17 min)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass


for step in range(max_steps):
    t0 = time.time() # just something lazy
    last_step = (step == max_steps-1)
    
    ##### (START) Large scale training code
    
    # Once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                loss = loss/ val_loss_steps
                val_loss_accum += loss.detach()
                
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step>0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model':raw_model.state_dict(),
                    'config':raw_model.config,
                    'steo':step,
                    'val_loss': val_loss_accum.item()
                }
                # Yu might also want to add optimizer.state_dict() and rng seeds etc.., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)
                
    # once in a while evaliate hellaswag
    if (step % 250 ==0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue 
            # Render the examples into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # Get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype = torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
            
        # Reduce the stats across all process
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

            
            
    ##### (END) Large Scale training code
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x,y = train_loader.next_batch()
        x,y = x.to(device) , y.to(device)
        with torch.autocast(device_type=device, dtype = torch.bfloat16):
            logits, loss = model(x,y)
            # print(logits.dtype) # torch.bfloat16 Activations are in BF16
            # print("model.transformer.wte.weight.dtype", model.transformer.wte.weight.dtype) # torch.float32 THIS IS STILL FLOAT32 not BF32/16
            # WHat gets converted to what is not that clear in pytorch, not all layers wont convert to BF16 such as layernorm, softmax.... etc 
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
        ##### STARTED GPU Stuff
        # import code; code.interact(local=locals())
        """
            >>> logits.dtype
            torch.float32
            - by default in pytorch tensors are stored in F32 when they are created. same case for all the activations, paramaters....
            - Thats a very high memory, way too much. 
            - 
        """
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)    
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Adding new utility function here to clip the gradients. Calculating the global norms of the parameters. generally added after loss.backward() only. Norm will be high in the beginning but then as the tranining continues it gets stablizes and value sgets below 1 and this is normal. 
    # Determine and set learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups: # this is the way to set learning rate in pytorch
        param_group['lr'] = lr
    optimizer.step()
    if device_type == 'cuda':    
        torch.cuda.synchronize() # for GPU, optional. this will make the GPU to wait for all the tasks scheduled by CPU before this line to run and then do what it is supposed to do 
    t1 = time.time()
    dt = (t1-t0) # Time diff in seconds # *1000 # time differenct in milliseconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    # tokens_per_sec = (train_loader.B * train_loader.T)/(t1-t0)
    ## (START) Added for large scale training
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
    ## (END) Added for large scale training       
    # print(f"loss: {loss_accum.item():.6f} ,lr: {lr:.4e}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}, norm: {norm:.4f}, step: {step:4d}")

if ddp:
    destroy_process_group()
    
# import sys; sys.exit(0)

#####-----------------------------------
# ## BEFORE OPTIMIZING TIME - 1200ms
# # in this very first loop before optimizing the model speed time taken was 1200ms
# train_loader = DataLoaderLite(B=16, T=1024)

# # Get logits 
# model = GPT(GPTConfig)
# model.to(device)

# # Optimize - 
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for i in range(50):
#     t0 = time.time() 
#     x,y = train_loader.next_batch()
#     x,y = x.to(device) , y.to(device)
#     optimizer.zero_grad()
#     logits, loss = model(x,y)
#     loss.backward()
#     optimizer.step()
#     t1 = time.time()
#     dt = (t1-t0)*1000 # time differenct in milliseconds
#     tokens_per_sec = (train_loader.B * train_loader.T)/(t1-t0)
#     print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
        
# import sys; sys.exit(0)
#####-----------------------------------

##### 9 ENDED HERE
##---------------------------------------------------------------------------------------------------------
# ##### 6 Continuation - changed few main running code blocks
# num_return_sequences = 5
# max_length = 30
# model = GPT.from_pretrained('gpt2')
# model.eval()
# model.to('cuda') # moving the whole model to GPU from CPU
# model.to(device)
# print("Didn't crash yet!")

##------------------------------------------------------------------------------
# ##### 7. PREFIX Tokens Creation started here below. 
# # so the plan is to make GPT 2 generate a text for a prefix text, that is 
# """Hello, I'm a language model,...""" 
# # Get a data Batch
# import tiktoken # Library from OpenAI
# enc = tiktoken.get_encoding('gpt2') # encoding that is developed for GPT2
# tokens = enc.encode("Hello, I'm a language model,")
# """
#     The above line will Encode this text to get a list of integer tokens which will look like this: 
#     15496, 11, 314, 1101, 257, 3303, 2746, 11   # [Shape = (8,)]
#     - got these from this app: https://tiktokenizer.vercel.app/?model=gpt2 for "Hello, I'm a language model,"
# """
# tokens = torch.tensor(tokens, dtype = torch.long) # (8, ), those texts that were encoded and converted to list will be tensorized here of 8 tokens hence shape (8, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1) # (5,8) # now we are repilicating those 8 tokens in the previous step for 5 times, to get 5 rows of 8 tokens which will be the initial input x that will live on GPU 
# x = tokens.to('cuda')

##------------------------------------------------------------------------------


##------------------------------------------------------------------------------
# # Generate rightg now, x's shape is (B, T), where B=5, T=8
# # set the seed to  42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1)<max_length:
#     # Forward the model to get the logits 
#     with torch.no_grad():
#         # import IPython ; IPython.embed() ; exit(1)
#         logits, _ = model(x) # (B,T,Vocab_size) # Added ", _" it was not there in the tutorial by Mr. Karpathy
#         # print("Passed logits")
#             # Take the logits at the last position
#         logits = logits[:,-1,:] # (B, vocab_size)
#             # Get the probabilities
#         probs = F.softmax(logits, dim=-1)
#             # Do top K sampling of 50 (As hugging face kept this as default)
#             # all the probabilities after top 50 will be equated to 0
#             # topK_probs here becomes (5,50), topK_indinces is (5,50)
#         topK_probs, topK_indinces = torch.topk(probs, 50, dim=-1)
#             # Select a token from the top-k probabilities
#         ix = torch.multinomial(topK_probs, 1) # (B, 1)
#             # Gather the corresponding indices
#         xcol = torch.gather(topK_indinces, -1, ix) # (B,1)
#             # append to the sequence
#         x = torch.cat((x,xcol), dim=1)

# # print the generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">", decoded) 
    
###### 6 END
##------------------------------------------------------------------------------


