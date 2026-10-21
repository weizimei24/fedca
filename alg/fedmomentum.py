import torch
import os

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record
from alg.flexlora import product_svd


def add_args(parser):
    parser.add_argument('--residual_threshold', type=float, default=parser.get_default('residual_threshold') if parser.get_default('residual_threshold') is not None else 0.9999,
                        help='Energy threshold τ for residual component selection (default: 0.9999)')
    return parser.parse_args()


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        return super().run(model)


class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.tau = getattr(args, 'residual_threshold', 0.9999)

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        r = self.args.lora_rank
        data_sum = sum([len(c.dataset['train']) for c in self.sampled_clients])
        if data_sum <= 0:
            raise ValueError('FedMomentum requires sampled clients with training rows')

        lora_a_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]
        new_global_lora = {k: v.clone() for k, v in self.global_lora.items()}

        for a_key in lora_a_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')

            # Exact thin SVD of sum_i w_i B_i A_i through the small QR core.
            # No dense update or random sketch is allocated.
            factors_a, factors_b = [], []
            for client in self.sampled_clients:
                w_i = len(client.dataset['train']) / data_sum
                A_i = client.lora[a_key].float()
                B_i = client.lora[b_key].float()
                factors_a.append(w_i * A_i)
                factors_b.append(B_i)
            U, S, Vt = product_svd(torch.cat(factors_b, dim=1), torch.cat(factors_a, dim=0))
            d, k = U.shape[0], Vt.shape[1]

            # ── Step 3: determine residual rank s via energy criterion ────────────
            total_energy = (S ** 2).sum()
            cumulative_energy = torch.cumsum(S ** 2, dim=0)
            # smallest r_eff such that cumulative energy ≥ τ · total_energy
            above = (cumulative_energy >= self.tau * total_energy).nonzero(as_tuple=True)[0]
            r_eff = ((above[0].item() + 1) if len(above) > 0 else len(S)) if total_energy.item() > 0 else 0
            # clamp r_eff to available singular values
            r_eff = min(r_eff, len(S))
            actual_r = min(r, r_eff, len(S))
            s = max(0, r_eff - actual_r)

            # ── Step 4: major components → new LoRA (balanced Σ^½ split) ─────────
            S_r = S[:actual_r]
            U_r = U[:, :actual_r]
            Vt_r = Vt[:actual_r, :]
            sqrt_S_r = torch.sqrt(S_r)

            # B = U_r · diag(Σ_r^½)   shape: (d, r)
            # A = diag(Σ_r^½) · V_r^T shape: (r, k)
            new_B = torch.zeros(d, r, device=U.device, dtype=U.dtype)
            # Keep directions for unused components. Zeroing BOTH factors
            # makes their gradients zero and permanently prevents learning.
            new_A = self.global_lora[a_key].float().clone()
            new_B[:, :actual_r] = U_r * sqrt_S_r
            new_A[:actual_r, :] = sqrt_S_r[:, None] * Vt_r

            orig_dtype = self.global_lora[a_key].dtype
            new_global_lora[b_key] = new_B.to(orig_dtype)
            new_global_lora[a_key] = new_A.to(orig_dtype)

            # ── Step 5: residual components → merge into backbone ─────────────────
            if s > 0 and base_key in self.model.state_dict():
                r_end = min(actual_r + s, len(S))
                S_s = S[actual_r:r_end]
                U_s = U[:, actual_r:r_end]
                Vt_s = Vt[actual_r:r_end, :]
                # W_residual = U_s · diag(Σ_s) · V_s^T
                W_residual = torch.matmul(U_s * S_s.unsqueeze(0), Vt_s)
                self.model.state_dict()[base_key].data.add_(
                    (W_residual * (self.args.lora_alpha / self.args.lora_rank)).to(self.model.state_dict()[base_key].dtype)
                )

        self.global_lora = new_global_lora
        self.model.load_state_dict(self.global_lora, strict=False)

    def save_adapter(self):
        super().save_adapter()
        from utils.method_state import export_state
        torch.save(export_state(self)['base'], os.path.join(self._adapter_path(), 'merged_weights.pt'))
