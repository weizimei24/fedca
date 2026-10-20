"""Project ILoRA: shared QR initialization and concatenated QR aggregation.

This follows the base ILoRA QR-initialization/aggregation path at a homogeneous
LoRA rank. It does not implement heterogeneous ranks or ILoRA-S control
variates and is not a full reproduction of the paper's experimental setting.
"""

import os

import torch

from alg.ftbase import FTBaseClient, FTBaseServer


class Client(FTBaseClient):
    pass


def _product_qr(stacked_b, stacked_a, rank):
    """First rank QR factors of B@A through a thin intermediate product."""
    b = stacked_b.float()
    a = stacked_a.float()
    if b.ndim != 2 or a.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError('ILoRA concatenated factors have incompatible shapes')
    if b.shape[1] > min(b.shape[0], a.shape[1]):
        # Tiny or very high-rank layers cannot use the thin-core shortcut.
        q, r = torch.linalg.qr(b @ a, mode='reduced')
    else:
        qb, rb = torch.linalg.qr(b, mode='reduced')
        qc, r = torch.linalg.qr(rb @ a, mode='reduced')
        q = qb @ qc
    if rank > q.shape[1]:
        raise ValueError('ILoRA server rank exceeds the available QR dimension')
    return q[:, :rank], r[:rank, :]


class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        if args.lora_rank <= 0:
            raise ValueError('ILoRA requires a positive LoRA rank')
        self._initialize_shared_qr()

    def _initialize_shared_qr(self):
        """Initialize all clients in one QR subspace, preserving W0 initially.

        The first r columns of QR(W0) are QR(W0[:, :r]); the matching first
        r rows of R are Q_r.T @ W0. This avoids a full matrix QR per client.
        Keep W0 - scale*B0*A0 as the global base in every round, otherwise
        the first aggregation would add the initialization back to W0.
        """
        state = self.model.state_dict()
        scaling = self.args.lora_alpha / self.args.lora_rank
        a_keys = [key for key in self.global_lora if 'lora_A' in key]
        if not a_keys:
            raise ValueError('No ILoRA layers found to initialize')
        with torch.no_grad():
            for a_key in a_keys:
                b_key = a_key.replace('lora_A', 'lora_B')
                base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
                if b_key not in state or base_key not in state:
                    raise KeyError(f'Missing ILoRA factor or base layer for {a_key}')
                rank = state[a_key].shape[0]
                base = state[base_key]
                if rank > min(base.shape):
                    raise ValueError(f'ILoRA rank exceeds dimensions of {base_key}')
                original = base.float()
                q, _ = torch.linalg.qr(original[:, :rank], mode='reduced')
                init_b = q.to(state[b_key].dtype)
                init_a = (q.T @ original).to(state[a_key].dtype)
                if init_a.shape != state[a_key].shape or init_b.shape != state[b_key].shape:
                    raise ValueError(f'ILoRA initialization shape mismatch for {a_key}')
                compensated = original - scaling * (init_b.float() @ init_a.float())
                base.copy_(compensated.to(base.dtype))
                state[a_key].copy_(init_a)
                state[b_key].copy_(init_b)
        self.global_lora = {
            key: value.clone() for key, value in state.items() if 'lora_' in key
        }
        print(f'ILoRA initialized {len(a_keys)} layers in a shared QR subspace')

    def aggregate(self):
        """QR-compress the data-weighted sum of client LoRA products."""
        total = sum(len(client.dataset['train']) for client in self.sampled_clients)
        if total <= 0:
            raise ValueError('ILoRA requires sampled clients with training rows')
        state = self.model.state_dict()
        updated = {}
        for a_key in (key for key in self.global_lora if 'lora_A' in key):
            b_key = a_key.replace('lora_A', 'lora_B')
            if b_key not in self.global_lora:
                raise KeyError(f'Missing ILoRA B factor for {a_key}')
            a_list, b_list = [], []
            for client in self.sampled_clients:
                if a_key not in client.lora or b_key not in client.lora:
                    raise KeyError(f'Client missing ILoRA factors for {a_key}')
                a, b = client.lora[a_key], client.lora[b_key]
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                    raise ValueError(f'ILoRA factor shape mismatch for {a_key}')
                weight = len(client.dataset['train']) / total
                a_list.append(weight * a.float())
                b_list.append(b.float())
            stacked_a = torch.cat(a_list, dim=0)
            stacked_b = torch.cat(b_list, dim=1)
            new_b, new_a = _product_qr(stacked_b, stacked_a, self.args.lora_rank)
            if new_a.shape != state[a_key].shape or new_b.shape != state[b_key].shape:
                raise ValueError(f'ILoRA aggregation shape mismatch for {a_key}')
            updated[a_key] = new_a.to(state[a_key].dtype)
            updated[b_key] = new_b.to(state[b_key].dtype)
        if not updated:
            raise ValueError('No ILoRA layers found to aggregate')
        self.global_lora = updated
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'ILoRA QR-aggregated {len(updated) // 2} layers')

    def save_adapter(self):
        # Initial QR compensation lives in the base weights, not in A/B.
        super().save_adapter()
        state = self.model.state_dict()
        base_keys = {
            key.replace('.lora_A.default.weight', '.base_layer.weight')
            for key in self.global_lora if 'lora_A' in key
        }
        compensated = {key: state[key].detach().cpu().clone() for key in base_keys}
        torch.save(compensated, os.path.join(self._adapter_path(), 'merged_weights.pt'))
        print(f'Saved {len(compensated)} ILoRA QR-compensated base layers')
