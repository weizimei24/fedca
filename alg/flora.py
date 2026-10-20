import os

import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        return super().run(model)

class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        data_total = sum(len(client.dataset['train']) for client in self.sampled_clients)
        if data_total <= 0:
            raise ValueError('FLoRA requires sampled clients with training rows')
        scaling = self.args.lora_alpha / self.args.lora_rank
        model_state = self.model.state_dict()
        pairs = []
        for a_key in (key for key in self.global_lora if 'lora_A' in key):
            b_key = a_key.replace('lora_A', 'lora_B')
            base_key = a_key.replace('.lora_A.default.weight', '.base_layer.weight')
            if b_key not in self.global_lora or base_key not in model_state:
                raise KeyError(f'Missing FLoRA factor or base layer for {a_key}')
            for client in self.sampled_clients:
                if a_key not in client.lora or b_key not in client.lora:
                    raise KeyError(f'Client missing FLoRA factors for {a_key}')
                a, b = client.lora[a_key], client.lora[b_key]
                if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]
                        or (b.shape[0], a.shape[1]) != model_state[base_key].shape):
                    raise ValueError(f'FLoRA update shape mismatch for {base_key}')
            pairs.append((a_key, b_key, base_key))
        if not pairs:
            raise ValueError('No FLoRA layers found to aggregate')
        with torch.no_grad():
            for a_key, b_key, base_key in pairs:
                a_list = []
                b_list = []
                for client in self.sampled_clients:
                    weight = len(client.dataset['train']) / data_total
                    a_list.append(weight * client.lora[a_key].float())
                    b_list.append(client.lora[b_key].float())
                stacked_a = torch.cat(a_list, dim=0)
                stacked_b = torch.cat(b_list, dim=1)
                delta = (stacked_b @ stacked_a) * scaling
                if delta.shape != model_state[base_key].shape:
                    raise ValueError(f'FLoRA update shape mismatch for {base_key}')
                model_state[base_key].add_(delta.to(model_state[base_key].dtype))
        # local_run() restored fresh rank-r adapters after every client;
        # merged base weights now carry the global update.
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f'FLoRA merged {len(pairs)} weighted updates into base weights')

    def save_adapter(self):
        # The generic saver only stores LoRA factors; FLoRA also changes W.
        super().save_adapter()
        state = self.model.state_dict()
        base_keys = {
            key.replace('.lora_A.default.weight', '.base_layer.weight')
            for key in self.global_lora if 'lora_A' in key
        }
        merged = {key: state[key].detach().cpu().clone() for key in base_keys}
        torch.save(merged, os.path.join(self._adapter_path(), 'merged_weights.pt'))
        print(f'Saved {len(merged)} FLoRA merged base layers')
