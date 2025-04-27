#!/usr/bin/env python3
import argparse
import pickle
import math

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core.frozen_dict import unfreeze
import numpy as np
import matplotlib.pyplot as plt

from util_gol import frame_to_tokens
from conway_lib import ConwayGame
from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    img_size: int
    block_size: int
    token_dim: int
    num_tokens: int
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    config: GPTConfig
    def setup(self):
        c = self.config
        assert c.n_embd % c.n_head == 0
        self.head_size = c.n_embd // c.n_head
        self.n_head = c.n_head
        self.c_attn = nn.Dense(c.n_embd * 3)
        self.c_proj = nn.Dense(c.n_embd)
        self.attn_dropout = nn.Dropout(c.dropout)
        self.resid_dropout = nn.Dropout(c.dropout)

    def __call__(self, x, *, train: bool):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_size).transpose(0,2,1,3)
        k = k.reshape(B, T, self.n_head, self.head_size).transpose(0,2,1,3)
        v = v.reshape(B, T, self.n_head, self.head_size).transpose(0,2,1,3)
        idx = jnp.arange(T)
        frame_idx = idx // self.config.num_tokens
        mask = (frame_idx[None, :] <= frame_idx[:, None]).reshape(1,1,T,T)
        att = (q @ k.transpose(0,1,3,2)) / math.sqrt(self.head_size)
        att = jnp.where(mask, att, float("-inf"))
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not train)
        y = att @ v
        y = y.transpose(0,2,1,3).reshape(B, T, C)
        return self.resid_dropout(self.c_proj(y), deterministic=not train)

class MLPBlock(nn.Module):
    config: GPTConfig
    def setup(self):
        c = self.config
        self.c_fc = nn.Dense(4 * c.n_embd)
        self.c_proj = nn.Dense(c.n_embd)
        self.dropout = nn.Dropout(c.dropout)

    def __call__(self, x, *, train: bool):
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x, deterministic=not train)

class Block(nn.Module):
    config: GPTConfig
    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLPBlock(self.config)

    def __call__(self, x, *, train: bool):
        x = x + self.attn(self.ln_1(x), train=train)
        x = x + self.mlp(self.ln_2(x), train=train)
        return x

class GPT(nn.Module):
    config: GPTConfig
    def setup(self):
        c = self.config
        self.token_proj = nn.Dense(c.n_embd)
        self.wpe = nn.Embed(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.h = [Block(c) for _ in range(c.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5)
        self.head = nn.Dense(c.token_dim)

    def __call__(self, tokens, *, train: bool):
        B, T, d = tokens.shape
        x = self.token_proj(tokens)
        pos = jnp.arange(T)[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        for blk in self.h:
            x = blk(x, train=train)
        x = self.ln_f(x)
        return self.head(x), None

    def get_token_latents(self, tokens, *, train=False, layer: Optional[int]=None, apply_ln=True):
        x = self.token_proj(tokens)
        pos = jnp.arange(tokens.shape[1])[None, :]
        x = x + self.wpe(pos)
        x = self.drop(x, deterministic=not train)
        for i, blk in enumerate(self.h):
            x = blk(x, train=train)
            if layer is not None and i == layer:
                break
        if apply_ln:
            x = self.ln_f(x)
        return x  # (B, T, n_embd)

def encode_frame(frame,img_size,patches,model,params):
    grid=(patches,patches)
    tokens=frame_to_tokens(frame[...,None],grid)
    lat=model.apply({'params':params},jnp.array(tokens[None]),method=GPT.get_token_latents)
    return np.array(lat[0])

def decode_latents(latents,img_size,patches,model,params):
    x=jnp.array(latents[None])
    logits=model.apply({'params':params},x,method=lambda m,x,train=False:m.head(m.ln_f(x)),train=False)[0]
    bits=(np.array(logits)>0).astype(int)
    patch_size=img_size//patches
    patches_arr=bits.reshape(patches**2,patch_size,patch_size)
    frame=np.zeros((img_size,img_size),int); idx=0
    for i in range(patches):
      for j in range(patches):
        frame[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]=patches_arr[idx]; idx+=1
    return frame

def simulate_count(frame,steps,toroidal):
    game=ConwayGame(width=frame.shape[0],height=frame.shape[1],grid_size=1,toroidal=toroidal)
    game.grid=(frame==1)
    history=game.run(num_iterations=steps)
    return history[-1].sum()

# ── REINFORCE training ─────────────────────────────────────────────────────────
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--patches", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.1,
                   help="fixed policy std dev over latents")
    p.add_argument("--lr", type=float, default=1e-2,
                   help="learning rate for μ-update")
    p.add_argument("--target", type=int, default=10)
    args=p.parse_args()

    # load model checkpoint
    ckpt=pickle.load(open(args.checkpoint,"rb")); ckpt=unfreeze(ckpt)
    true_block=ckpt["wpe"]["embedding"].shape[0]
    num_tokens=args.patches**2
    token_dim=(args.img_size//args.patches)**2
    config=GPTConfig(args.img_size,true_block,token_dim,num_tokens)
    model=GPT(config)
    rng=jax.random.PRNGKey(0)
    fake=jnp.zeros((1,true_block,num_tokens))
    variables=model.init(rng,fake,train=False)
    params=ckpt

    # initialize policy mean μ randomly
    T, D = true_block, num_tokens*0  # placeholder to get shape
    # get shape from one encode
    dummy_frame=(np.random.rand(args.img_size,args.img_size)<0.5).astype(int)
    latent0=encode_frame(dummy_frame,args.img_size,args.patches,model,params)
    T, D = latent0.shape
    mu = np.zeros((T,D),dtype=np.float32)

    for gen in range(1,args.iters+1):
        # 1) sample a batch of latents: z_i = μ + σ * ε_i
        eps = np.random.randn(args.pop, T, D)
        Z   = mu[None] + args.sigma * eps

        # 2) compute rewards r_i = -|count − target|
        rewards = np.zeros(args.pop, dtype=np.float32)
        for i in range(args.pop):
            f0 = decode_latents(Z[i],args.img_size,args.patches,model,params)
            c  = simulate_count(f0,args.steps,True)
            rewards[i] = -abs(c - args.target)

        # 3) baseline and advantage
        baseline = rewards.mean()
        adv = rewards - baseline

        # 4) REINFORCE update on μ:
        #    ∇_μ log N(z;μ,σ²I) = (z - μ)/σ²
        grad_mu = (adv[:,None,None] * (Z - mu[None]))/(args.sigma**2)
        mu += args.lr * grad_mu.mean(axis=0)

        # 5) log
        print(f"[gen {gen:4d}] reward  min={rewards.min():.2f}"
              f" mean={rewards.mean():.2f} max={rewards.max():.2f}")

        # 6) visualize first 10 trajectories
        n_display = min(10, args.pop)
        histories = []
        for i in range(n_display):
            f0 = decode_latents(Z[i],args.img_size,args.patches,model,params)
            game=ConwayGame(width=args.img_size,height=args.img_size,grid_size=1,toroidal=True)
            game.grid=(f0==1)
            hist=game.run(num_iterations=args.steps)
            frames=[f0]+[h.astype(int) for h in hist]
            histories.append(frames)

        n_cols = args.steps+1
        fig,axes=plt.subplots(n_display,n_cols,
                              figsize=(n_cols*1.5,n_display*1.5))
        for r in range(n_display):
            for c in range(n_cols):
                ax=axes[r,c]
                ax.imshow(histories[r][c],cmap='gray',interpolation='nearest')
                ax.axis('off')
                if r==0: ax.set_title(f"Step {c}")
        plt.tight_layout()
        plt.savefig(f"reinforce_gen_{gen:04d}.png",bbox_inches='tight',pad_inches=0.1)
        plt.close(fig)

if __name__=="__main__":
    main()
