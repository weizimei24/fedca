"""FlexLoRA product-space aggregation and rank-r SVD redistribution."""

import torch

from alg.ftbase import FTBaseClient, FTBaseServer


def product_svd(stacked_b, stacked_a):
    """SVD of B@A through a small core, without materializing a dense update."""
    qb, rb = torch.linalg.qr(stacked_b.float(), mode='reduced')
    qa, ra = torch.linalg.qr(stacked_a.float().T, mode='reduced')
    u, sigma, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    return qb @ u, sigma, vh @ qa.T


def _truncated_product_svd(stacked_b, stacked_a, rank):
    u, sigma, vh = product_svd(stacked_b, stacked_a)
    if rank > sigma.numel():
        raise ValueError('FlexLoRA rank exceeds the available product dimension')
    new_b = u[:, :rank] * sigma[:rank]
    new_a = vh[:rank, :]
    return new_b, new_a


class Client(FTBaseClient):
    pass


class Server(FTBaseServer):
    def aggregate(self):
        total = sum(len(client.dataset['train']) for client in self.sampled_clients)
        if total <= 0:
            raise ValueError('FlexLoRA requires sampled clients with training rows')
        updated = {}
        state = self.model.state_dict()
        for a_key in (key for key in self.global_lora if 'lora_A' in key):
            b_key = a_key.replace('lora_A', 'lora_B')
            if b_key not in self.global_lora:
                raise KeyError(f'Missing FlexLoRA B factor for {a_key}')
            a_list, b_list = [], []
            for client in self.sampled_clients:
                if a_key not in client.lora or b_key not in client.lora:
                    raise KeyError(f'Client missing FlexLoRA factors for {a_key}')
                a, b = client.lora[a_key], client.lora[b_key]
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                    raise ValueError(f'FlexLoRA factor shape mismatch for {a_key}')
                weight = len(client.dataset['train']) / total
                a_list.append(weight * a.float())
                b_list.append(b.float())
            stacked_a = torch.cat(a_list, dim=0)
            stacked_b = torch.cat(b_list, dim=1)
            new_b, new_a = _truncated_product_svd(stacked_b, stacked_a, self.args.lora_rank)
            if new_a.shape != state[a_key].shape or new_b.shape != state[b_key].shape:
                raise ValueError(f'FlexLoRA redistribution shape mismatch for {a_key}')
            # PEFT applies alpha/r to both local and redistributed adapters.
            updated[a_key] = new_a.to(state[a_key].dtype)
            updated[b_key] = new_b.to(state[b_key].dtype)
        if not updated:
            raise ValueError('No FlexLoRA layers found to aggregate')
        self.global_lora = updated
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'FlexLoRA redistributed {len(updated) // 2} rank-{self.args.lora_rank} layers')
