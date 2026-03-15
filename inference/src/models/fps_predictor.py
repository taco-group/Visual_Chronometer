import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange, repeat
from src.models.autoencoder2plus1d_1dcnn import AutoencoderKL2plus1D_1dcnn
from src.modules.attention_temporal_videoae import CrossAttention
from utils.common_utils import instantiate_from_config
import math

class FPSPredictor(pl.LightningModule):
    def __init__(
        self,
        ddconfig, # Encoder config
        ppconfig, # Temporal Encoder config (if separate) or integrated
        lossconfig, # Not used for VAE loss, but maybe for consistency
        embed_dim,
        use_quant_conv=True,
        ckpt_path=None,
        freeze_encoder=True,
        hidden_dim=1024, # Dimension for the probe token and MLP
        input_key="video",
        monitor="val/loss",
        logdir=None,
        warmup_steps=2000,
        n_layers=4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = 1e-4
        self.freeze_encoder = freeze_encoder
        self.input_key = input_key
        self.logdir = logdir
        self.warmup_steps = warmup_steps
        self.n_layers = n_layers
            
        # 1. Instantiate the VAE model to get the encoder
        # We reuse the existing VAE class to easily load weights
        # 1. Instantiate the VAE model to get the encoder
        # We reuse the existing VAE class to easily load weights
        self.vae = AutoencoderKL2plus1D_1dcnn(
            ddconfig=ddconfig,
            ppconfig=ppconfig,
            lossconfig=lossconfig, # Dummy or real, doesn't matter as we won't use its loss
            embed_dim=embed_dim,
            use_quant_conv=use_quant_conv,
            ckpt_path=ckpt_path,
        )
        
        # We only need the encoder parts
        if self.freeze_encoder:
            self.vae.eval()
            self.vae.freeze()
        else:
            # Delete unused decoder parts to avoid DDP hanging
            # These parts are initialized but not used in forward(), causing DDP to wait for their gradients
            if hasattr(self.vae, "decoder"):
                del self.vae.decoder
            if hasattr(self.vae, "decoder_temporal"):
                del self.vae.decoder_temporal
            if hasattr(self.vae, "post_quant_conv"):
                del self.vae.post_quant_conv
            if hasattr(self.vae, "encoder_temporal"):
                del self.vae.encoder_temporal
        
        self.feat_dim = 2 * ddconfig["z_channels"] if use_quant_conv else ddconfig["z_channels"] 
        self.probe_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Project encoder output to hidden_dim if necessary
        self.proj_in = nn.Linear(self.feat_dim, hidden_dim)
        
        # Attention Pooling
        if n_layers == 1:
            self.attn_pool = CrossAttention(
                query_dim=hidden_dim,
                context_dim=hidden_dim,
                heads=8,
                dim_head=64,
                dropout=0.1
            )
        else:
            self.attn_pool = nn.ModuleList([
                CrossAttention(
                    query_dim=hidden_dim,
                    context_dim=hidden_dim,
                    heads=8,
                    dim_head=64,
                    dropout=0.1
                ) for _ in range(n_layers)
            ])
        
        # 3. MLP for Regression
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Output: log(FPS) or just FPS. Plan says log(FPS)
        )
        
    def forward(self, x):
        # x: [B, C, T, H, W]
        
        # 1. Encode
        if self.freeze_encoder:
            with torch.no_grad():
                # encoder(x) returns raw moments [B, 2*z, t, h, w]
                latents = self.vae.encoder(x)
                if self.vae.use_quant_conv:
                    latents = self.vae.quant_conv(latents)
        else:
             latents = self.vae.encoder(x)
             if self.vae.use_quant_conv:
                 latents = self.vae.quant_conv(latents)
        
        # latents: [B, C_enc, T, H, W]
        b, c, t, h, w = latents.shape
        
        # Flatten: [B, T*H*W, C]
        latents = rearrange(latents, 'b c t h w -> b (t h w) c')
        
        # Project to hidden_dim
        latents = self.proj_in(latents) # [B, Seq, Hidden]
        
        # Probe Token
        probe = repeat(self.probe_token, '1 1 d -> b 1 d', b=b)
        
        # Cross Attention
        # query=probe, context=latents
        # Output: [B, 1, Hidden]
        pooled = probe
        if self.n_layers == 1:
            pooled = self.attn_pool(pooled, context=latents)
        else:
            for attn in self.attn_pool:
                pooled = attn(pooled, context=latents)
        
        # MLP
        # Output: [B, 1, 1] -> [B, 1]
        pred_log_fps = self.mlp(pooled).squeeze(-1)
        
        return pred_log_fps

    def training_step(self, batch, batch_idx):
        x = batch[self.input_key]
        fps_gt = batch['fps'] # [B]
        
        # Check shapes
        if len(fps_gt.shape) == 1:
            fps_gt = fps_gt.unsqueeze(1) # [B, 1]
            
        pred_log_fps = self(x) # [B, 1]
        
        log_fps_gt = torch.log(fps_gt + 1e-6)
        loss = torch.nn.functional.l1_loss(pred_log_fps, log_fps_gt)
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/mae_fps", torch.abs(torch.exp(pred_log_fps) - fps_gt).mean(), prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.input_key]
        fps_gt = batch['fps']
        
        if len(fps_gt.shape) == 1:
            fps_gt = fps_gt.unsqueeze(1)

        pred_log_fps = self(x)
        log_fps_gt = torch.log(fps_gt + 1e-6)
        loss = torch.nn.functional.l1_loss(pred_log_fps, log_fps_gt)
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/mae_fps", torch.abs(torch.exp(pred_log_fps) - fps_gt).mean(), prog_bar=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        if self.warmup_steps > 0:
            def warmup_fn(step):
                if step < self.warmup_steps:
                    return float(step) / float(max(1, self.warmup_steps))
                return 1.0
            
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_fn),
                "interval": "step",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        return log
