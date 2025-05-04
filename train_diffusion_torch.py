#!/usr/bin/env python3
import os, csv, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import torchvision.utils as vutils

# ─── Dataset of consecutive frame‑pairs ─────────────────────────────────────────
class FramePairsDataset(Dataset):
    def __init__(self, csv_file, img_size, num_frames):
        self.pairs = []
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith('State'): continue
                assert len(row) == num_frames, f"Expected {num_frames} cols"
                frames = []
                for cell in row:
                    s = cell.strip()
                    assert len(s) == img_size*img_size
                    arr = np.frombuffer(s.encode('ascii'), np.uint8) - 48
                    frames.append(arr.reshape(img_size, img_size).astype(np.float32))
                for i in range(len(frames)-1):
                    self.pairs.append((frames[i], frames[i+1]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        x0, x1 = self.pairs[idx]
        return torch.from_numpy(x0)[None], torch.from_numpy(x1)[None]

# ─── Sinusoidal time embedding ─────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:,None].float() * freq[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

# ─── Residual block ────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, channels*2)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x, t):
        h = self.norm1(x); h = F.silu(h); h = self.conv1(h)
        scale, shift = self.time_mlp(t).chunk(2, dim=-1)
        h = h * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
        h = self.norm2(h); h = F.silu(h); h = self.conv2(h)
        return x + h

# ─── “channel‑reducing” up‑block ───────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.project = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.res     = ResBlock(out_ch, time_dim)
    def forward(self, x, t):
        x = self.project(x)
        return self.res(x, t)

# ─── U‑Net with correct up/down bookkeeping, conditioned on x0 ────────────────
class UNet2D(nn.Module):
    def __init__(self, base_ch=64, depth=3, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )
        # now accept 2 channels: noisy x1_t and conditioning x0
        self.init_conv = nn.Conv2d(2, base_ch, 3, padding=1)

        # channel counts per level
        self.level_ch = [base_ch * (2**i) for i in range(depth+1)]

        # down path
        self.down_blocks = nn.ModuleList()
        self.down_pools  = nn.ModuleList()
        self.down_proj   = nn.ModuleList()
        for idx, ch in enumerate(self.level_ch[:-1]):
            self.down_blocks.append(ResBlock(ch, base_ch))
            self.down_pools .append(nn.AvgPool2d(2))
            self.down_proj  .append(nn.Conv2d(ch, self.level_ch[idx+1], 1))

        # bottleneck
        bot_ch = self.level_ch[-1]
        self.bottleneck = ResBlock(bot_ch, base_ch)

        # up path
        self.up_trans  = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for idx in range(depth, 0, -1):
            in_ch  = self.level_ch[idx]
            out_ch = self.level_ch[idx-1]
            self.up_trans .append(nn.ConvTranspose2d(in_ch,  out_ch, 4, stride=2, padding=1))
            self.up_blocks.append(UpBlock(2*out_ch, out_ch, time_dim=base_ch))

        self.final_norm = nn.GroupNorm(8, base_ch)
        self.final_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        h = self.init_conv(x)
        skips = []
        for blk, pool, proj in zip(self.down_blocks, self.down_pools, self.down_proj):
            h = blk(h, t); skips.append(h); h = pool(h); h = proj(h)
        h = self.bottleneck(h, t)
        for trans, blk in zip(self.up_trans, self.up_blocks):
            skip = skips.pop()
            h = trans(h)
            h = torch.cat([h, skip], dim=1)
            h = blk(h, t)
        h = self.final_norm(h); h = F.silu(h)
        logits = self.final_conv(h)
        # squash into [0,1] range
        return torch.sigmoid(logits)

def q_sample(x, t, alphas):
    noise = torch.randn_like(x)
    a = alphas[t].view(-1,1,1,1)
    return a.sqrt()*x + (1-a).sqrt()*noise, noise

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train_csv",  default="conway_train.csv")
    p.add_argument("--val_csv",    default="conway_val.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps",      type=int, default=5000)
    p.add_argument("--log_every",  type=int, default=100)
    p.add_argument("--img_size",   type=int, default=16)
    p.add_argument("--num_frames", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="gol_diffusion_pytorch", config=vars(args))

    train_ds = FramePairsDataset(args.train_csv, args.img_size, args.num_frames)
    val_ds   = FramePairsDataset(args.val_csv,   args.img_size, args.num_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = UNet2D().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse   = nn.MSELoss()
    betas  = torch.linspace(1e-4, 2e-2, 1000, device=device)
    alphas = torch.cumprod(1 - betas, dim=0)

    os.makedirs("viz", exist_ok=True)
    step=0
    while step<args.steps:
        for x0,x1 in train_loader:
            x0,x1 = x0.to(device), x1.to(device)
            t = torch.randint(0,1000,(x0.size(0),),device=device)
            xt,noise = q_sample(x1, t, alphas)
            inp = torch.cat([xt, x0], dim=1)
            pred = model(inp, t)  # now in [0,1]
            loss = mse(pred, noise)

            opt.zero_grad(); loss.backward(); opt.step()
            step+=1

            if step % args.log_every == 0:
                model.eval(); vl=0.0
                with torch.no_grad():
                    for vx0,vx1 in val_loader:
                        clean = vx1.to(device)
                        vt = torch.randint(0,1000,(clean.size(0),),device=device)
                        vxt, vn = q_sample(clean, vt, alphas)
                        inp_v = torch.cat([vxt, vx0.to(device)], dim=1)
                        out   = model(inp_v, vt)
                        vl += mse(out, vn).item()
                vl /= len(val_loader)
                print(f"[{step:5d}] train={loss.item():.6f} val={vl:.6f}")
                wandb.log({"train/loss":loss.item(),"val/loss":vl},step=step)

                # visualize 10 GT vs pred
                count=0
                with torch.no_grad():
                    for vx0,vx1 in val_loader:
                        clean = vx1.to(device)
                        vt = torch.randint(0,1000,(clean.size(0),),device=device)
                        vxt, vn = q_sample(clean, vt, alphas)
                        inp_v = torch.cat([vxt, vx0.to(device)], dim=1)
                        out   = model(inp_v, vt)
                        for i in range(clean.shape[0]):
                            if count>=10: break
                            gt = clean[i]
                            pd = out[i]  # already in [0,1]
                            binary_pd = (pd > 0.5).float()
                            pair = torch.cat([gt, binary_pd], dim=2)
                            vutils.save_image(pair, f"viz/{count:02d}.png", normalize=False)
                            count+=1
                        if count>=10: break
                model.train()
            if step>=args.steps: break

    os.makedirs("ckpts", exist_ok=True)
    torch.save(model.state_dict(), "ckpts/diffusion_pytorch.pt")
    print("Saved ckpts/diffusion_pytorch.pt")

if __name__=="__main__":
    main()
