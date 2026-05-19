"""
VF Value Synthesis - Inference Module

Medical AI research prototype for Visual Field (VF) prediction from fundus images.
This module provides model loading and single-image inference functionality.

RESEARCH PROTOTYPE NOTICE:
This is a research prototype, not a diagnostic or production system.
Use by qualified ophthalmologists / medical professionals only.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Model and path configuration."""
    
    # Model dimensions
    VIT_MODEL = "vit_base_patch16_224"
    VIT_EMBED_DIM = 768
    CLINICAL_DIM = 4
    MLP_HIDDEN = 128
    MLP_EMBED_DIM = 128
    COND_DIM = 256
    VF_DIM = 61
    
    # Diffusion schedule
    T_STEPS = 1000
    BETA_START = 5e-5
    BETA_END = 0.01
    
    # Image preprocessing
    IMAGE_SIZE = 224
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


# ============================================================================
# Model Components
# ============================================================================

class FundusViTEncoder(nn.Module):
    """
    Vision Transformer encoder for fundus images.
    Returns the [CLS] token embedding (768-d).
    
    Optional: pass fundus_ckpt='RETFound_cfp_weights.pth' for retinal weights.
    Download from: https://github.com/rmaphoh/RETFound_MAE
    """
    
    def __init__(self, model_name=cfg.VIT_MODEL, pretrained=True, fundus_ckpt=None):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if fundus_ckpt and os.path.exists(fundus_ckpt):
            state = torch.load(fundus_ckpt, map_location="cpu")
            key = "model" if "model" in state else "state_dict"
            self.vit.load_state_dict(state.get(key, state), strict=False)
            print(f"[ViT] Loaded fundus checkpoint: {fundus_ckpt}")
    
    def forward(self, x):
        """Encode fundus image to 768-d embedding."""
        return self.vit(x)  # (B, 768)


class ClinicalMLP(nn.Module):
    """
    Processes clinical inputs (Age, Gender, IOP, CCT) into a 128-d embedding.
    
    Supports optional fields via learnable null tokens.
    For each of the 4 clinical fields, a learnable scalar null token is registered.
    When a field is missing (mask=False), its value is replaced by the learned null.
    """
    
    N_FIELDS = cfg.CLINICAL_DIM  # == 4  [Age, Gender, IOP, CCT]
    
    def __init__(self, in_dim=cfg.CLINICAL_DIM, hidden=cfg.MLP_HIDDEN,
                 out_dim=cfg.MLP_EMBED_DIM):
        super().__init__()
        
        # Pretrained MLP layers (weights loaded from checkpoint)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        
        # Per-field learnable null tokens (shape [4])
        # Initialized to 0.0 (== z-score mean)
        self.null_tokens = nn.Parameter(torch.zeros(self.N_FIELDS))
    
    def forward(self, x, mask=None):
        """
        Process clinical features.
        
        Parameters
        ----------
        x    : FloatTensor [B, 4]  z-scored clinical values
        mask : BoolTensor  [B, 4]  True = field is PRESENT, False = field is MISSING
               If None, all fields are treated as present (legacy behaviour).
        
        Returns
        -------
        FloatTensor [B, 128]
        """
        if mask is not None:
            # null_tokens [4] -> broadcast to [B, 4]
            null = self.null_tokens.unsqueeze(0).expand_as(x)
            x = torch.where(mask, x, null)
        return self.net(x)  # [B, 128]


class ConditioningFusion(nn.Module):
    """Fuse ViT and MLP embeddings into conditioning vector."""
    
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cfg.VIT_EMBED_DIM + cfg.MLP_EMBED_DIM, cfg.COND_DIM),
            nn.LayerNorm(cfg.COND_DIM),
            nn.GELU(),
        )
    
    def forward(self, vit_emb, mlp_emb):
        """Fuse embeddings into 256-d conditioning vector."""
        return self.proj(torch.cat([vit_emb, mlp_emb], dim=-1))


class DiffusionSchedule:
    """Diffusion process schedule (forward + reverse)."""
    
    def __init__(self, T=cfg.T_STEPS, beta_start=cfg.BETA_START, beta_end=cfg.BETA_END):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_1mab = torch.sqrt(1.0 - alpha_bar)
    
    def to(self, device):
        """Move schedule tensors to device."""
        for attr in ("betas", "alphas", "alpha_bar", "sqrt_ab", "sqrt_1mab"):
            setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def q_sample(self, x0, t, noise=None):
        """Forward diffusion q(x_t|x_0): add noise at step t."""
        if noise is None:
            noise = torch.randn_like(x0)
        s = self.sqrt_ab[t].unsqueeze(-1)
        sm = self.sqrt_1mab[t].unsqueeze(-1)
        return s * x0 + sm * noise, noise


class SinusoidalTimeEmbed(nn.Module):
    """Sinusoidal time embedding for diffusion step."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-np.log(10000) *
                         torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock1D(nn.Module):
    """Residual block conditioned on time + clinical conditioning."""
    
    def __init__(self, channels, time_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, channels))
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels))
    
    def forward(self, x, t_emb, cond):
        h = self.fc1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb) + self.cond_proj(cond)
        h = self.fc2(F.silu(self.norm2(h)))
        return x + h


class VFDenoiser(nn.Module):
    """
    Denoiser network for VF synthesis.
    Predicts noise at diffusion step t.
    
    Input  : (B,61) x_t  +  (B,) t  +  (B,256) cond
    Output : (B,61) predicted noise
    """
    
    def __init__(self, vf_dim=cfg.VF_DIM, cond_dim=cfg.COND_DIM,
                 hidden=256, depth=6, time_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(time_dim),
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.input_proj = nn.Linear(vf_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden, time_dim, cond_dim) for _ in range(depth)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, vf_dim))
    
    def forward(self, x_t, t, cond):
        t_emb = self.time_embed(t)
        h = self.input_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        return self.output_proj(h)


class CDPMVFSynthesizer(nn.Module):
    """
    Conditional Diffusion Probabilistic Model for Visual Field synthesis.
    
    Encodes fundus image + optional clinical features → generates VF field
    using DDPM reverse sampling with optional-field support.
    """
    
    def __init__(self, fundus_ckpt=None):
        super().__init__()
        self.vit = FundusViTEncoder(fundus_ckpt=fundus_ckpt)
        self.mlp = ClinicalMLP()  # with learnable null_tokens
        self.fusion = ConditioningFusion()
        self.denoiser = VFDenoiser()
    
    def encode(self, images, clinical, clin_mask=None):
        """
        Fuse fundus image + clinical embeddings → conditioning vector [B, 256].
        
        Parameters
        ----------
        images     : FloatTensor [B, 3, H, W]
        clinical   : FloatTensor [B, 4]  z-scored values
        clin_mask  : BoolTensor  [B, 4]  True = present, False = use null token
                                          If None, all fields are present.
        
        Returns
        -------
        FloatTensor [B, 256]
        """
        vit_emb = self.vit(images)  # [B, 768]
        mlp_emb = self.mlp(clinical, mask=clin_mask)  # [B, 128]
        return self.fusion(vit_emb, mlp_emb)  # [B, 256]
    
    def forward(self, images, clinical, xt, t, clin_mask=None):
        """Predict noise at diffusion step t."""
        return self.denoiser(xt, t, self.encode(images, clinical, clin_mask))


# ============================================================================
# Inference Utilities
# ============================================================================

def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load pretrained CDPM model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to best_model.pt
    device : torch.device
    
    Returns
    -------
    model : CDPMVFSynthesizer
    stats : dict with normalization statistics (clin_mean, clin_std, vf_mean, vf_std)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create and load model
    model = CDPMVFSynthesizer()
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(device)
    model.eval()
    
    # Extract normalization stats
    stats = {
        "clin_mean": np.array(ckpt["clin_mean"], dtype=np.float32),
        "clin_std": np.array(ckpt["clin_std"], dtype=np.float32),
        "vf_mean": np.array(ckpt["vf_mean"], dtype=np.float32),
        "vf_std": np.array(ckpt["vf_std"], dtype=np.float32),
    }
    
    return model, stats


def preprocess_image(image_path_or_pil, image_size=cfg.IMAGE_SIZE):
    """
    Preprocess fundus image for model input.
    
    Parameters
    ----------
    image_path_or_pil : str or PIL.Image
        Path to image file or PIL Image object
    image_size : int
        Target image size (224 x 224)
    
    Returns
    -------
    img_tensor : FloatTensor [1, 3, 224, 224]
    """
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")
    
    return tfm(img).unsqueeze(0)


def prepare_clinical_tensor(age, gender, iop, cct, clin_mean, clin_std):
    """
    Prepare z-scored clinical feature tensor.
    
    Parameters
    ----------
    age : float or None
    gender : str ("M" or "F") or None
    iop : float or None
    cct : float or None
    clin_mean : np.array [4]
    clin_std : np.array [4]
    
    Returns
    -------
    clin_tensor : FloatTensor [1, 4]
    clin_mask : BoolTensor [1, 4]  True = field provided, False = missing
    """
    # Normalize gender to 0.0 (M) or 1.0 (F)
    gender_val = 1.0 if (gender and str(gender).strip().upper().startswith('F')) else 0.0
    
    # Create raw and mask arrays
    raw = np.array([
        float(age) if age is not None else 0.0,
        gender_val,
        float(iop) if iop is not None else 0.0,
        float(cct) if cct is not None else 0.0,
    ], dtype=np.float32)
    
    mask = np.array([
        age is not None,
        gender is not None,
        iop is not None,
        cct is not None,
    ], dtype=bool)
    
    # Z-score (only for fields that are provided)
    raw_z = (raw - clin_mean) / clin_std
    
    clin_tensor = torch.tensor(raw_z, dtype=torch.float32).unsqueeze(0)
    clin_mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    
    return clin_tensor, clin_mask


@torch.no_grad()
def ddpm_reverse_sample(model, schedule, cond, device, n_steps=300):
    """
    DDPM reverse sampling: x_T ~ N(0,I) → x_0 (synthesised VF, z-score space).
    
    Parameters
    ----------
    model : VFDenoiser or CDPMVFSynthesizer
    schedule : DiffusionSchedule
    cond : FloatTensor [B, COND_DIM]  conditioning vector
    device : torch.device
    n_steps : int  number of reverse steps (default 300)
    
    Returns
    -------
    x : FloatTensor [B, VF_DIM]  synthesised VF in z-score space
    """
    x = torch.randn(cond.size(0), cfg.VF_DIM, device=device)
    
    for t_idx in reversed(range(n_steps)):
        t_tensor = torch.full((cond.size(0),), t_idx, device=device, dtype=torch.long)
        pred_eps = model.denoiser(x, t_tensor, cond)
        
        beta_t = schedule.betas[t_idx]
        alpha_t = schedule.alphas[t_idx]
        alpha_bar_t = schedule.alpha_bar[t_idx]
        alpha_bar_prev = (schedule.alpha_bar[t_idx - 1] if t_idx > 0
                          else torch.tensor(1.0, device=device))
        
        # Estimate clean x_0
        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(-3.0, 3.0)
        
        # Posterior mean
        coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x
        
        if t_idx > 0:
            var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x = mean + torch.sqrt(var) * torch.randn_like(x)
        else:
            x = mean
    
    return x


def infer_single_vf(
    image_path_or_pil,
    checkpoint_path,
    age=None,
    gender=None,
    iop=None,
    cct=None,
    n_steps=300,
    device=None
):
    """
    Run single-image VF inference.
    
    Parameters
    ----------
    image_path_or_pil : str or PIL.Image
        Path to fundus image or PIL Image object
    checkpoint_path : str
        Path to best_model.pt
    age : float, optional
        Age in years
    gender : str, optional
        "M" (Male) or "F" (Female)
    iop : float, optional
        Intraocular pressure in mmHg
    cct : float, optional
        Central corneal thickness in micrometers
    n_steps : int
        Number of DDPM reverse steps (default 300)
    device : torch.device, optional
        Compute device (defaults to cfg.DEVICE)
    
    Returns
    -------
    vf_denorm : np.array [61]
        Synthesised VF values in real space (0-35 dB range, typically)
    vf_zscore : np.array [61]
        Synthesised VF in z-score space (before denormalization)
    """
    if device is None:
        device = torch.device(cfg.DEVICE)
    
    # Load model and stats
    model, stats = load_model_from_checkpoint(checkpoint_path, device)
    schedule = DiffusionSchedule().to(device)
    
    # Preprocess image
    img_tensor = preprocess_image(image_path_or_pil, image_size=cfg.IMAGE_SIZE)
    img_tensor = img_tensor.to(device)
    
    # Prepare clinical features
    clin_tensor, clin_mask = prepare_clinical_tensor(
        age, gender, iop, cct,
        stats["clin_mean"], stats["clin_std"]
    )
    clin_tensor = clin_tensor.to(device)
    clin_mask = clin_mask.to(device)
    
    # Encode
    with torch.no_grad():
        cond = model.encode(img_tensor, clin_tensor, clin_mask=clin_mask)
        
        # DDPM reverse sampling
        synth_norm = ddpm_reverse_sample(
            model, schedule, cond, device, n_steps=n_steps
        )
    
    # Denormalize to real VF space
    vf_mean_t = torch.tensor(stats["vf_mean"], device=device)
    vf_std_t = torch.tensor(stats["vf_std"], device=device)
    
    vf_denorm_t = synth_norm * vf_std_t + vf_mean_t
    
    # Convert to numpy
    vf_zscore = synth_norm.cpu().numpy().squeeze()
    vf_denorm = vf_denorm_t.cpu().numpy().squeeze()
    
    return vf_denorm, vf_zscore


if __name__ == "__main__":
    # Example usage (requires best_model.pt in current directory)
    print("VF Value Synthesis Inference Module")
    print("Loaded successfully. Use infer_single_vf() for predictions.")
