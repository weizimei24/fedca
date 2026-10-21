import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record
from alg.flexlora import _truncated_product_svd

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        client_model = model

        if not getattr(getattr(self, 'args', None), '_shared', False):
            for param in client_model.parameters():
                param.requires_grad = False
            for name, param in client_model.named_parameters():
                if "lora_B" in name:
                    param.requires_grad = True

        self.last_train_loss = self.trainer.train(client_model)
        self.lora = {k: v.clone() for k, v in client_model.state_dict().items() if 'lora_B' in k}

class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        r = self.args.lora_rank
        
        # 1. Aggregate LoRA_B First
        data_sum = sum([len(client.dataset['train']) for client in self.sampled_clients])
        from collections import defaultdict
        aggregated = defaultdict(lambda: 0)

        for client in self.sampled_clients:
            model = client.lora
            for k, v in model.items():
                if 'lora_B' in k: aggregated[k] = aggregated[k] + v * len(client.dataset['train']) / data_sum

        # 2. Generate SVD Decomposition for LoRA_A
        lora_keys = [k for k in self.global_lora.keys() if 'lora_A' in k]
        for a_key in lora_keys:
            b_key = a_key.replace('lora_A', 'lora_B')
            
            A = self.global_lora[a_key]
            B = aggregated[b_key]
            # Float32 QR/core SVD avoids unsupported fp16 SVD and does not
            # materialize a dense vocabulary/hidden-space update.
            new_b, new_a = _truncated_product_svd(B, A, r)
            self.global_lora[a_key] = new_a.to(A.dtype)
            self.global_lora[b_key] = new_b.to(B.dtype)
            
        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")
