"""Load a trained value net for use inside agents and labellers (CPU or GPU).

    from value_model import Value
    v = Value('oracle/value.pt')          # needs torch only
    v.of_game(game, side)                 # P(side wins); needs ts-env (features.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_value import ValueMLP  # noqa: E402


class Value:
    def __init__(self, path: str, device: str = 'cpu'):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.model = ValueMLP(ck['dim'], ck['hidden'], ck['depth']).to(device)
        self.model.load_state_dict(ck['state_dict']); self.model.eval()
        self.mean = ck['mean'].to(device); self.std = ck['std'].to(device); self.device = device
        self.feat_dim = ck.get('feat_dim', ck['dim'])
        self.feat_idx = torch.tensor(ck['feat_idx'], device=device) if ck.get('feat_idx') is not None else None
        self.aux = bool(ck.get('aux', False))  # whether input vectors carry the aux block (3081 + AUX_DIM)

    @torch.no_grad()
    def __call__(self, feats: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.asarray(feats, dtype=np.float32), device=self.device)
        if x.ndim == 1:
            x = x[None]
        x = x[:, self.feat_idx] if self.feat_idx is not None else x[:, :self.feat_dim]
        return torch.sigmoid(self.model((x - self.mean) / self.std)).cpu().numpy()

    def of_game(self, game, side) -> float:
        """P(side wins) from side's view; exact for a finished game (the encoding cannot tell who
        caused DEFCON 1, so terminal states are read from the rules, not from the net)."""
        if game.decision is None:
            w = game.state.winner
            return 0.5 if w is None else (1.0 if w is side else 0.0)
        from features import featurize  # imported here so loading the net needs only torch
        return float(self(featurize(game, side, aux=self.aux))[0])
