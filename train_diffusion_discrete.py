#!/usr/bin/env python3
import os, csv, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import torchvision.utils as vutils

# ─── Dataset ────────────────────────────────────────────────────────────────────
class FramePairsDataset(Dataset):
    def __init__(self, csv_file, img_size, num_frames, t_skip=0):
        self.pairs = []
        with open(csv_file) as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith("State"): continue
                frames = []
                for cell in row:
                    arr = (np.frombuffer(cell.strip().encode("ascii"), np.uint8) - 48
                           ).astype(np.float32)
                    frames.append(arr.reshape(img_size, img_size))
                for i in range(len(frames)-t_skip-1):
                    self.pairs.append((frames[i], frames[i+t_skip+1]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        x0, x1 = self.pairs[idx]
        return torch.from_numpy(x0)[None], torch.from_numpy(x1)[None]

# ─── Sinusoidal time embedding ─────────────────────────────────────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim): 
        super().__init__(); self.dim=dim
    def forward(self, t):
        half=self.dim//2
        freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
        args = t[:,None].float()*freq[None]
        return torch.cat([torch.sin(args),torch.cos(args)],-1)

# ─── Residual block ────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, c, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8,c)
        self.conv1 = nn.Conv2d(c,c,3,padding=1)
        self.time  = nn.Linear(time_dim, c*2)
        self.norm2 = nn.GroupNorm(8,c)
        self.conv2 = nn.Conv2d(c,c,3,padding=1)
    def forward(self,x,t):
        h=self.norm1(x); h=F.silu(h); h=self.conv1(h)
        scale,shift = self.time(t).chunk(2,-1)
        h = h*(1+scale[:,:,None,None]) + shift[:,:,None,None]
        h=self.norm2(h); h=F.silu(h); h=self.conv2(h)
        return x+h

# ─── Up‑block reduces 2·c→c then ResBlock ────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self,in_c,out_c,time_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_c,out_c,1)
        self.res  = ResBlock(out_c, time_dim)
    def forward(self,x,t):
        return self.res(self.proj(x), t)

# ─── U‑Net conditioned on x₀, outputs logits for 2 classes ─────────────────────
class UNet2D(nn.Module):
    def __init__(self, base_ch=64, depth=3, time_dim=64):
        super().__init__()
        # time embedding
        self.time = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, base_ch),
            nn.SiLU(),
            nn.Linear(base_ch, base_ch),
        )
        # input conv accepts (x_t and x₀) → 2 channels
        self.init = nn.Conv2d(2, base_ch, 3, padding=1)
        # down levels
        self.levels = [base_ch*(2**i) for i in range(depth+1)]
        self.down_b = nn.ModuleList()
        self.down_p = nn.ModuleList()
        self.down_q = nn.ModuleList()
        for i,c in enumerate(self.levels[:-1]):
            self.down_b.append(ResBlock(c, base_ch))
            self.down_p.append(nn.AvgPool2d(2))
            self.down_q.append(nn.Conv2d(c, self.levels[i+1],1))
        # bottleneck
        self.bot = ResBlock(self.levels[-1], base_ch)
        # up levels
        self.up_t = nn.ModuleList()
        self.up_b = nn.ModuleList()
        for i in range(depth,0,-1):
            inc = self.levels[i]
            outc= self.levels[i-1]
            self.up_t.append(nn.ConvTranspose2d(inc,outc,4,2,1))
            self.up_b.append(UpBlock(2*outc,outc, base_ch))
        # final
        self.norm = nn.GroupNorm(8,base_ch)
        self.out  = nn.Conv2d(base_ch,2,3,padding=1)  # 2-class logits

    def forward(self, x, t):
        t = self.time(t)
        h = self.init(x)
        skips=[]
        for blk,pool,q in zip(self.down_b,self.down_p,self.down_q):
            h=blk(h,t); skips.append(h); h=pool(h); h=q(h)
        h=self.bot(h,t)
        for trans,blk in zip(self.up_t,self.up_b):
            skip=skips.pop()
            h=trans(h)
            h=blk(torch.cat([h,skip],1),t)
        h=self.norm(h); h=F.silu(h)
        return self.out(h)  # logits shape [B,2,H,W]

# ─── discrete forward sampling q(x_t | x0) via bit‑flip schedule ─────────────
def q_sample_discrete(x0, t, alpha_bar):
    # x0 ∈{0,1}, alpha_bar[t] = prob of keep
    # p_t = P(x_t=1) = x0*alpha_bar + (1-x0)*(1-alpha_bar)
    # sample Bernoulli
    # b = alpha_bar[t].view(-1,1,1,1)
    # p = x0*b + (1-x0)*(1-b)
    # return torch.bernoulli(p)
    keep_prob = alpha_bar[t].view(-1,1,1,1).expand_as(x0)          # shape [B,1,1,1]
    keep_mask = torch.bernoulli(keep_prob)           # 1 = keep original
    rand_noise = torch.bernoulli(0.5 * torch.ones_like(x0))
    x_t = keep_mask * x0 + (1 - keep_mask) * rand_noise
    return x_t

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train_csv", default="conway_train.csv")
    p.add_argument("--val_csv",   default="conway_val.csv")
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--steps",     type=int,default=5000)
    p.add_argument("--log_every", type=int,default=100)
    p.add_argument("--img_size",  type=int,default=16)
    p.add_argument("--num_frames",type=int,default=20)
    p.add_argument("--t_skip",   type=int,default=0)
    args=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="gol_discrete",config=vars(args))

    # data
    train_ds=FramePairsDataset(args.train_csv,args.img_size,args.num_frames, args.t_skip)
    val_ds  =FramePairsDataset(args.val_csv,  args.img_size,args.num_frames, args.t_skip)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,drop_last=True)
    val_loader  =DataLoader(val_ds,  batch_size=args.batch_size,shuffle=True,drop_last=False)

    # model + optimizer + loss
    model=UNet2D().to(device)
    opt  =torch.optim.Adam(model.parameters(),lr=1e-4)
    ce   =nn.CrossEntropyLoss()

    # discrete schedule
    T=1000
    betas = torch.linspace(1e-4,0.02,T,device=device)
    alphas=1-betas
    alpha_bar=torch.cumprod(alphas,0)

    step=0
    while step<args.steps:
        for x0,x1 in train_loader:
            x0,x1 = x0.to(device),x1.to(device)
            # sample diffusion time
            t = torch.randint(0,T,(x0.size(0),),device=device)
            # corrupt x1 → x1_t
            x1_t = q_sample_discrete(x1, t, alpha_bar)
            # model input: [x1_t, x0]
            inp  = torch.cat([x1_t, x0],1)
            logits = model(inp, t)           # [B,2,H,W]
            # target is original x1 (class 0 or 1)
            loss = ce(logits, x1.long().squeeze(1))
            opt.zero_grad(); loss.backward(); opt.step()
            step+=1

            if step % args.log_every==0:
                # eval loss
                model.eval(); vl=0
                with torch.no_grad():
                    for vx0,vx1 in val_loader:
                        vx0,vx1=vx0.to(device),vx1.to(device)
                        t2=torch.randint(0,T,(vx0.size(0),),device=device)
                        vxt=q_sample_discrete(vx1,t2,alpha_bar)
                        inp2=torch.cat([vxt,vx0],1)
                        lgts=model(inp2,t2)
                        vl+=ce(lgts,vx1.long().squeeze(1)).item()
                vl/=len(val_loader)
                print(f"[{step}] train={loss.item():.4f} val={vl:.4f}")
                wandb.log({"train/ce":loss.item(),"val/ce":vl},step=step)

                # viz 10 random val
                os.makedirs("viz",exist_ok=True)
                cnt=0
                with torch.no_grad():
                    for vx0,vx1 in val_loader:
                        vx0,vx1=vx0.to(device),vx1.to(device)
                        t2=torch.randint(0,T,(vx0.size(0),),device=device)
                        print("t2 for this batch:", t2.tolist()[:10])
                        vxt=q_sample_discrete(vx1,t2,alpha_bar)
                        inp2=torch.cat([vxt,vx0],1)
                        lgts=model(inp2,t2)
                        probs=F.softmax(lgts,1)[:,1:]  # P(class=1)
                        pred=(probs>0.5).float()
                        for i in range(vx1.size(0)):
                            if cnt>=10: break
                            pair=torch.cat([vx0[i],vxt[i],vx1[i],pred[i]],2)
                            vutils.save_image(pair, f"viz/{cnt:02d}.png",normalize=False)
                            cnt+=1
                        if cnt>=10: break
                model.train()
            if step>=args.steps: break

    # save
    torch.save(model.state_dict(),"unet_discrete.pth")
    print("Done")

if __name__=="__main__":
    main()
