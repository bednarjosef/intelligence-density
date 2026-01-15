import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        d = x.shape[-1]
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos.unsqueeze(0).unsqueeze(2)) + (rotate_half(x) * sin.unsqueeze(0).unsqueeze(2))


class MaskedSelfAttention(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        assert CONFIG['n_embd'] % CONFIG['n_head'] == 0
        self.head_dim = CONFIG['n_embd'] // CONFIG['n_head']
        self.n_head = CONFIG['n_head']
        
        self.c_attn = nn.Linear(CONFIG['n_embd'], 3 * CONFIG['n_embd'], bias=False)
        self.c_proj = nn.Linear(CONFIG['n_embd'], CONFIG['n_embd'], bias=False)
        
        # RoPE
        self.rotary = RotaryPositionalEmbeddings(self.head_dim)
        
        # Causal Mask
        self.register_buffer("bias", torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size']))
                                     .view(1, 1, CONFIG['block_size'], CONFIG['block_size']))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape to [B, T, nh, hs]
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2) 

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    

class BidirectionalAttention(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        assert CONFIG['n_embd'] % CONFIG['n_head'] == 0
        self.head_dim = CONFIG['n_embd'] // CONFIG['n_head']
        self.n_head = CONFIG['n_head']
        
        self.c_attn = nn.Linear(CONFIG['n_embd'], 3 * CONFIG['n_embd'], bias=False)
        self.c_proj = nn.Linear(CONFIG['n_embd'], CONFIG['n_embd'], bias=False)
        
        # RoPE
        self.rotary = RotaryPositionalEmbeddings(self.head_dim)
        
        # Causal Mask
        self.register_buffer("bias", torch.tril(torch.ones(CONFIG['block_size'], CONFIG['block_size']))
                                     .view(1, 1, CONFIG['block_size'], CONFIG['block_size']))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape to [B, T, nh, hs]
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(q)
        q = apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_pos_emb(k, cos, sin).transpose(1, 2) 

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) <- without masking - see the whole text
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
    

class FeedForward(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['n_embd'], 4 * CONFIG['n_embd'], bias=False),
            nn.ReLU(),
            nn.Linear(4 * CONFIG['n_embd'], CONFIG['n_embd'], bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.ln1 = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.attn = MaskedSelfAttention(CONFIG)
        self.ln2 = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.ffwd = FeedForward(CONFIG)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.ln1 = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.attn = BidirectionalAttention(CONFIG)
        self.ln2 = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.ffwd = FeedForward(CONFIG)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, CONFIG, recursion_depth):
        super().__init__()
        self.CONFIG = CONFIG
        self.recursion_depth = recursion_depth
        
        self.token_embedding_table = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        
        self.transformer = Transformer(CONFIG)
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.lm_head = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'], bias=False)

        # weight tying
        self.token_embedding_table.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # embed only tokens
        x = self.token_embedding_table(idx)
        
        # recursive passes
        for _ in range(self.recursion_depth):
            x = self.transformer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens=None, greedy=False):
        if max_new_tokens is None:
            max_new_tokens = self.CONFIG['block_size']
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.CONFIG['block_size']:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if greedy:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Stop generation if we hit EOT (Optional but good for speed)
            # if idx_next.item() == self.CONFIG['eot_token_id']:
            #     break
        return idx
    

class EmbeddingGPT(nn.Module):
    def __init__(self, CONFIG, recursion_depth):
        super().__init__()
        self.CONFIG = CONFIG
        self.recursion_depth = recursion_depth
        self.token_embedding_table = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        self.transformer = Transformer(CONFIG)
        self.lm_head = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'], bias=False)

    def forward(self, inputs_embeds=None, targets=None):
        x = inputs_embeds

        # Run the transformer
        for _ in range(self.recursion_depth):
            x = self.transformer(x)
            
        logits = self.lm_head(self.transformer.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            
        return logits, loss
    

class Extractor(nn.Module):
    def __init__(self, CONFIG, recursion_depth):
        super().__init__()
        self.CONFIG = CONFIG
        self.recursion_depth = recursion_depth
        
        self.token_embedding = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        self.encoder = BidirectionalTransformer(CONFIG) 
        
        self.head_1 = Transformer(CONFIG)
        self.head_2 = Transformer(CONFIG)
        
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.lm_head_1 = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'], bias=False)
        self.lm_head_2 = nn.Linear(CONFIG['n_embd'], CONFIG['vocab_size'], bias=False)

        self.token_embedding.weight = self.lm_head_1.weight
        self.lm_head_1.weight = self.lm_head_2.weight

    def forward_head(self, context_embeds, idx_generated, head_transformer, head_out_layer):
        gen_embeds = self.token_embedding(idx_generated)
        
        x = torch.cat([context_embeds, gen_embeds], dim=1)
        
        for _ in range(self.recursion_depth):
            x = head_transformer(x)
            
        x = self.ln_f(x)
        logits = head_out_layer(x)
        return logits

    def forward(self, src_idx, tgt1_idx, tgt2_idx):
        x_enc = self.token_embedding(src_idx)
        for _ in range(self.recursion_depth):
            x_enc = self.encoder(x_enc)
        
        logits_1 = self.forward_head(x_enc, tgt1_idx, self.head_1, self.lm_head_1)
        logits_2 = self.forward_head(x_enc, tgt2_idx, self.head_2, self.lm_head_2)

        return logits_1, logits_2

    def generate(self, src_idx, max_new_tokens=20, start_token_id=0):
        B, T = src_idx.shape

        context_embeds = self.token_embedding(src_idx)
        for _ in range(self.recursion_depth):
            context_embeds = self.encoder(context_embeds)

        curr_idx_1 = torch.full((B, 1), start_token_id, dtype=torch.long, device=src_idx.device)
        
        for _ in range(max_new_tokens):
            logits = self.forward_head(context_embeds, curr_idx_1, self.head_1, self.lm_head_1)
            
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            curr_idx_1 = torch.cat((curr_idx_1, idx_next), dim=1)

        curr_idx_2 = torch.full((B, 1), start_token_id, dtype=torch.long, device=src_idx.device)
        
        for _ in range(max_new_tokens):
            logits = self.forward_head(context_embeds, curr_idx_2, self.head_2, self.lm_head_2)
            
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            curr_idx_2 = torch.cat((curr_idx_2, idx_next), dim=1)
            
        return curr_idx_1[:, 1:], curr_idx_2[:, 1:]


class Decider(nn.Module):
    def __init__(self, CONFIG, recursion_depth, num_resolvers):
        super().__init__()
        self.CONFIG = CONFIG
        self.recursion_depth = recursion_depth
        
        self.token_embedding_table = nn.Embedding(CONFIG['vocab_size'], CONFIG['n_embd'])
        
        self.transformer = BidirectionalTransformer(CONFIG)
        self.ln_f = nn.LayerNorm(CONFIG['n_embd'], bias=False)
        self.classifier_head = nn.Linear(CONFIG['n_embed'], num_resolvers + 1, bias=False)  # last one is END
        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) 
        
        for _ in range(self.recursion_depth):
            x = self.transformer(x)
            
        x = self.ln_f(x)
        # final_token_vector = x[:, -1, :]
        final_vector = x.mean(dim=1)
        logits = self.classifier_head(final_vector)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss


class Resolver(nn.Module):
    def __init__(self, CONFIG, recursion_depth):
        super().__init__()
        self.CONFIG = CONFIG
        self.recursion_depth = recursion_depth
        # should be essentially a GPT() wrapper...


class RouterBlock(nn.Module):
    def __init__(self, CONFIG):
        super().__init__()
        self.CONFIG = CONFIG
        self.num_resolvers = 4

        self.decider = Decider(CONFIG, recursion_depth=4, num_resolvers=self.num_resolvers)
        self.resolvers = [GPT(CONFIG, recursion_depth=4) for _ in range(self.num_resolvers)]

    def forward(self, idx, targets=None):
        aggregate_output = None

        # router loop
        while True:
            decider_logits, _decider_loss = self.decider.forward(idx)
            decider_probabilities = F.softmax(decider_logits, dim=1)
            predicted_resolver_idx = torch.argmax(decider_probabilities, dim=1)

            # end
            if predicted_resolver_idx == self.num_resolvers:
                break
            
            current_resolver = self.resolvers[predicted_resolver_idx]
            current_resolver_output = current_resolver.generate(idx)

            next_input, result = Extractor.generate(current_resolver_output, max_new_tokens=256)

            aggregate_output += result
            idx = next_input

        logits = aggregate_output
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss