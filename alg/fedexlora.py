"""FedEx-LoRA: exact weighted update via a residual in the base weights."""

import os

import torch

from alg.ftbase import FTBaseClient, FTBaseServer


class Client(FTBaseClient):
    pass


class Server(FTBaseServer):
    def aggregate(self):
        total = sum(len(client.dataset['train']) for client in self.sampled_clients)
        if total <= 0:
            raise ValueError('FedEx-LoRA requires sampled clients with training rows')

        model_state = self.model.state_dict()
        updated = dict(self.global_lora)
        scaling = self.args.lora_alpha / self.args.lora_rank
        pairs = []
        for a_key in (key for key in self.global_lora if 'lora_A' in key):
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
            if b_key not in self.global_lora or base_key not in model_state:
                raise KeyError(f'Missing FedEx-LoRA factor or base layer for {a_key}')
            for client in self.sampled_clients:
                if a_key not in client.lora or b_key not in client.lora:
                    raise KeyError(f'Client missing FedEx-LoRA factors for {a_key}')
                a, b = client.lora[a_key], client.lora[b_key]
                if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
                        or (b.shape[0], a.shape[1]) != model_state[base_key].shape):
                    raise ValueError(f'FedEx-LoRA update shape mismatch for {base_key}')
            pairs.append((a_key, b_key, base_key))
        if not pairs:
            raise ValueError('No FedEx-LoRA layers found to aggregate')

        # W + scale * (sum p_k B_k A_k - B_bar A_bar) + scale * B_bar A_bar
        # equals the exact weighted average of the client effective models.
        with torch.no_grad():
            for a_key, b_key, base_key in pairs:
                mean_a = sum(
                    (len(client.dataset['train']) / total) * client.lora[a_key].float()
                    for client in self.sampled_clients
                )
                mean_b = sum(
                    (len(client.dataset['train']) / total) * client.lora[b_key].float()
                    for client in self.sampled_clients
                )
                mean_product = sum(
                    (len(client.dataset['train']) / total)
                    * (client.lora[b_key].float() @ client.lora[a_key].float())
                    for client in self.sampled_clients
                )
                updated[a_key] = mean_a.to(model_state[a_key].dtype)
                updated[b_key] = mean_b.to(model_state[b_key].dtype)
                # Compensate the actual stored factors after dtype rounding.
                residual = (
                    mean_product - updated[b_key].float() @ updated[a_key].float()
                ) * scaling
                model_state[base_key].add_(residual.to(model_state[base_key].dtype))

        self.global_lora = updated
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'FedEx-LoRA merged {len(pairs)} residual updates into base weights')

    def save_adapter(self):
        # The generic saver only stores A and B, but the residual is in W.
        super().save_adapter()
        state = self.model.state_dict()
        base_keys = {
            key.replace('.lora_A.default.weight', '.base_layer.weight')
            for key in self.global_lora if 'lora_A' in key
        }
        merged = {key: state[key].detach().cpu().clone() for key in base_keys}
        torch.save(merged, os.path.join(self._adapter_path(), 'merged_weights.pt'))
        print(f'Saved {len(merged)} FedEx-LoRA residual-adjusted base layers')
