import torch
import torch.nn as nn

class MLA(nn.Module):
    '''
    Q = h @ Wq

    # compress
    c_kv = h @ W_dkv

    # cache compressed latent
    latent_cache.append(c_kv)

    # reconstruct when needed
    K = c_kv @ W_uk
    V = c_kv @ W_uv
    '''
    def __init__(
            self,
            hidden_dim,
            num_heads,
            head_dim,
            latent_dim
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim

        # Query Projection

        self.w_q = nn.Linear(
            in_features=hidden_dim,
            out_features=num_heads * head_dim,
            bias=False
        )

        # Down Projection
        # compress hidden -> latent

        self.w_dkv = nn.Linear(
            in_features = hidden_dim,
            out_features = latent_dim,
            bias = False
        )

        # Up Projection
        # latent -> Keys

        self.w_uk = nn.Linear(
            in_features = latent_dim,
            out_features = num_heads * head_dim,
            bias = False
        )

        # latent -> Values

        self.w_uv = nn.Linear(
            in_features = latent_dim,
            out_features = num_heads * head_dim,
            bias = False

        )

        # RoPE
        self.rope = nn.Linear(
            in_features=head_dim,
            out_features=head_dim,
            bias=False
        )

        # Mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(num_heads, head_dim), diagonal=1).bool()
        )

        # out
        self.out_proj = nn.Linear(
            in_features= num_heads * head_dim,
            out_features= hidden_dim
        )

    def forward(self, h):

        B, T, _ = h.shape

        # Queries

        q = self.w_q(h)
        c_kv = self.w_dkv(h) # Compress KV

        latent_cache = c_kv 

        # Reconstruct k & v
        k = self.w_uk(c_kv)
        v = self.w_uv(c_kv)

        # [B, T, H, D]
        q = q.view(B, T,
            self.num_heads,
            self.head_dim
        )
        k = k.view(B, T,
            self.num_heads,
            self.head_dim
        )
        v = v.view(B, T,
            self.num_heads,
            self.head_dim
        )

        # Move head dimension before sequence dimension [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (
            q @ k.transpose(-2, -1)
        ) * (self.head_dim ** -0.5)

        mask = torch.triu(
            torch.ones(T, T, device=h.device),
            diagonal=1
        ).bool()

        attn = attn.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(B, T, -1)

        out = self.out_proj(out)

        return out, latent_cache


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        return self.net(x
        )

class DeepSeekMoe(nn.Module):
    def __init__(
            self,
            dim,
            hidden_dim,
            num_experts = 256,
            top_k = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # shared expert
        self.shared_expert = Expert(dim, hidden_dim)

        # routed experts
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_experts)]
        )

        # Router centroids
        self.router = nn.Parameter(
            torch.randn(num_experts, dim) # learnable array b = [b_1, b_2, ..., b_256]_(num_experts, dim)
        )

        # Load balancing biases
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts) # [0,0,0...,num_experts]
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        flat_x = x.reshape(-1, D) # [B*T , D]

        # shared output
        shared_out = self.shared_expert(flat_x)

        # raw routing scores (s_n)
        scores = torch.sigmoid(
            flat_x @ self.router.T
        )

        # Only Model selection
        # Bias-adjusted routing
        routing_scores = (
            scores + self.expert_bias
        )

        # Top-k selection
        topk_scores, topk_idx = torch.topk(
            routing_scores,
            self.top_k,
            dim=-1
        )

        # IMPORTANT:
        # Use ORIGINAL scores, not biased scores | [weights]
        gather_scores = torch.gather(
            scores,
            1,
            topk_idx
        )
        gather_scores = nn.functional.softmax(
            gather_scores,
            dim=-1
        )

        # Expert aggregation
        moe_out = torch.zeros_like(flat_x)

        for k in range(self.top_k):

            expert_ids = topk_idx[:, k]

            for expert_id in range(self.num_experts):

                mask = (
                    expert_ids == expert_id
                )

                if mask.sum() == 0:
                    continue

                selected = flat_x[mask]

                expert_out = self.experts[
                    expert_id
                ](selected)

                weight = gather_scores[
                    mask,
                    k
                ].unsqueeze(-1)

                moe_out[mask] += (
                    weight * expert_out
                )

        # Combine shared + routed
        out = shared_out + moe_out

        out = out.view(B, T, D)

        return out


class MTPModule(nn.Module):

    def __init__(
            self, dim, vocab_size
    ):
        super().__init__()

        self.proj = nn.Linear(
            dim * 2,
            dim
        )
        self.block = TransformerBlock(dim)
        self.head = nn.Linear(
            dim,
            vocab_size
        )

    def forward(
            self,
            hidden_state,
            next_token_embedding
    ):
        x = torch.cat(
            (hidden_state, next_token_embedding),
            dim=-1
        )
        x = self.proj(x)
        h, _ = self.block(x)
        logits = self.head(h)
        
        return h, logits


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
    
'''
The parameter values used here are for demonstration purposes and have been chosen randomly.
You can customize them according to your requirements.
'''

class TransformerBlock(nn.Module):
    
    def __init__(
            self,
            dim,
            num_heads = 4,
            head_dim = 64,
            latent_dim = 128,
            moe_hidden = 128,
            num_experts=256,
            top_k=8
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)

        self.attn = MLA(
            hidden_dim= dim,
            num_heads = num_heads,
            head_dim = head_dim,
            latent_dim = latent_dim
        )

        self.norm2 = RMSNorm(dim)

        self.moe = DeepSeekMoe(
            dim = dim,
            hidden_dim = moe_hidden,
            num_experts = num_experts,
            top_k = top_k
        )

    def forward(self, x):

        """
        x
        │
        ├── RMSNorm
        ├── MLA Attention
        ├── Residual Add
        │
        ├── RMSNorm
        ├── DeepSeekMoE
        ├── Residual Add
        │
      output
        """

        attn_out, cache = self.attn(
            self.norm1(x)
        )

        x = x + attn_out

        moe_out = self.moe(
            self.norm2(x)
        )

        x = x + moe_out

        return x, cache
    

class DeepSeekV3(nn.Module):

    def __init__(self, vocab_size, dim, num_layers = 2):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, dim)

        # Transformer Blocks (Main Model)
        self.layers = nn.ModuleList([
            TransformerBlock(dim=dim) for _ in range(num_layers)
        ])

        # Output Head
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # MTP Module (Predicts the NEXT next token)
        self.mtp_module = MTPModule(dim, vocab_size)
        # Weight sharing from paper: head weights are shared
        self.mtp_module.head.weight = self.head.weight

    def forward(self, input_ids):
        # input_ids: [Batch, Seq_len]
        x = self.embedding(input_ids)

        caches = []
        for layer in self.layers:
            x, cache = layer(x)
            caches.append(cache)
        
        main_hidden = self.norm(x)
        main_logits = self.head(main_hidden) # [B, T, Vocab_Size]

        return main_hidden, main_logits, caches

    def forward_mtp(self, main_hidden, target_ids):
        # We need the embeddings of the TARGET tokens for MTP
        # target_ids: [Batch, Seq_Len]
        target_emb = self.embedding(target_ids)

        # MTP predicts based on current hidden state + NEXT actual token's embedding
        mtp_hidden, mtp_logits = self.mtp_module(main_hidden, target_emb)
        return mtp_logits


"""
Single training iteration with Main LM loss + MTP loss
""" 

import torch.nn.functional as F

device = "cuda"

model = DeepSeekV3(
    vocab_size=32000, dim=512
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4
)


# example batch [B, T]
input_ids = torch.randint( 0, 32000,
    (4, 128)).to(device)

model.train()

# FORWARD PASS
main_hidden, main_logits, caches = model(input_ids)

# MAIN LM LOSS (predicts next token)
main_logits = main_logits[:, :-1, :]
main_targets = input_ids[:, 1:]

main_loss = F.cross_entropy(
    main_logits.reshape(-1, main_logits.size(-1)),
    main_targets.reshape(-1)
)

# MTP LOSS
# input embeddings for MTP (token t+1 embeddings)
mtp_input_ids = input_ids[:, 1:-1]

# targets are token t+2
mtp_targets = input_ids[:, 2:]

# hidden states aligned with t
mtp_hidden = main_hidden[:, :-2, :]


mtp_logits = model.forward_mtp(
    mtp_hidden,
    mtp_input_ids
)

mtp_loss = F.cross_entropy(
    mtp_logits.reshape(-1, mtp_logits.size(-1)),
    mtp_targets.reshape(-1)
)

# TOTAL LOSS
loss = main_loss + 0.3 * mtp_loss

# BACKPROP
optimizer.zero_grad()

loss.backward()

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    1.0
)

optimizer.step()

print("main loss:", main_loss.item())
print("mtp loss :", mtp_loss.item())
print("total    :", loss.item())