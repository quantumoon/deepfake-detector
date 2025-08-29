import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import AUROC, F1Score


class Backbone(nn.Module):
    """
    RGB branch → Swin‑Tiny,
    FFT branch → EfficientNet‑B0.
    """
    def __init__(self, num_types: int) -> None:
        super().__init__()
        # RGB branch (Swin‑Tiny)
        self.rgb = timm.create_model("swin_tiny_patch4_window7_224",
                                     pretrained=True,
                                     in_chans=3,
                                     num_classes=0)
        self._freeze_rgb()
        rgb_dim = self.rgb.num_features

        # FFT branch (Eff‑B0)
        self.freq = timm.create_model("efficientnet_b0",
                                      pretrained=True,
                                      in_chans=1,
                                      num_classes=0,
                                      global_pool="avg")
        self._freeze_freq()
        freq_dim = self.freq.num_features

        # Classification Heads
        self.feat_dim = rgb_dim + freq_dim
        self.head_bin = nn.Linear(self.feat_dim, 1)
        self.head_type = nn.Linear(self.feat_dim, num_types)

    def _freeze_rgb(self) -> None:
        for n, p in self.rgb.named_parameters():
            if n.startswith("norm"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def _freeze_freq(self) -> None:
        freeze_p = True
        for n, p in self.freq.named_parameters():
            if freeze_p and n.startswith("blocks.5"):
                freeze_p = False
            p.requires_grad = not freeze_p

    def forward(self,
                x_rgb: torch.Tensor,
                x_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        f_rgb = self.rgb(x_rgb)  # (B, rgb_dim)
        f_freq = self.freq(x_freq)  # (B, freq_dim)
        feat = torch.cat([f_rgb, f_freq], dim=1)
        logit_bin = self.head_bin(feat).squeeze(1)
        logit_type = self.head_type(feat)
        return logit_bin, logit_type


class LitDeepfakeDetector(pl.LightningModule):
    def __init__(self,
                 num_types: int = 7,
                 lr: float = 3e-4,
                 weight_decay: float = 5e-4,
                 lambda_type: float = 0.3,
                 bin_type: float = 0.7):
        
        super().__init__()
        self.save_hyperparameters()

        self.model = Backbone(num_types)
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.train_f1 = F1Score(task="multiclass", num_classes=num_types, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_types, average="macro")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return {"optimizer": opt, "lr_scheduler": sched}

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor):
        return self.model(rgb, freq)

    # training
    def training_step(self, batch, batch_idx):
        rgb, freq, y_bin, y_type = batch
        logit_bin, logit_type = self(rgb, freq)
        loss_bin = self.bce(logit_bin, y_bin)
        loss_type = self.ce(logit_type, y_type)
        loss = self.hparams.bin_type * loss_bin + self.hparams.lambda_type * loss_type

        preds_bin = torch.sigmoid(logit_bin)
        preds_type = torch.softmax(logit_type, dim=1)
        self.train_auc.update(preds_bin, y_bin.int())
        self.train_f1.update(preds_type, y_type)

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        auc = self.train_auc.compute()
        f1 = self.train_f1.compute()
        self.log("train/roc_auc", auc, prog_bar=True)
        self.log("train/f1_macro", f1, prog_bar=True)
        self.train_auc.reset()
        self.train_f1.reset()

    # validation
    def validation_step(self, batch, batch_idx):
        rgb, freq, y_bin, y_type = batch
        logit_bin, logit_type = self(rgb, freq)
        loss_bin = self.bce(logit_bin, y_bin)
        loss_type = self.ce(logit_type, y_type)
        preds_bin = torch.sigmoid(logit_bin)
        preds_type = torch.softmax(logit_type, dim=1)

        self.val_auc.update(preds_bin, y_bin.int())
        self.val_f1.update(preds_type, y_type)

        loss = self.hparams.bin_type * loss_bin + self.hparams.lambda_type * loss_type
        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        f1 = self.val_f1.compute()
        self.log("val/roc_auc", auc, prog_bar=True)
        self.log("val/f1_macro", f1, prog_bar=True)
        self.val_auc.reset()
        self.val_f1.reset()

    # prediction
    def predict_step(self, batch, batch_idx):
        rgb, freq = batch
        logit_bin, _ = self(rgb, freq)
        return torch.sigmoid(logit_bin)

    # FLOPs helper
    def flops(self, img_size: int = 224) -> Optional[float]:
        try:
            from thop import profile
            self.eval()
            device = next(self.parameters()).device
            rgb = torch.zeros(1, 3, img_size, img_size, device=device)
            freq = torch.zeros(1, 1, img_size, img_size, device=device)
            flops, _ = profile(self.model, inputs=(rgb, freq), verbose=True)
            return flops / 1e9
        except ImportError:
            warnings.warn("Install 'thop' for FLOPs computation.")
            return None