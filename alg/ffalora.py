"""FFA-LoRA: keep the initialized A fixed and aggregate only trained B."""

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                param.requires_grad = True
        self.last_train_loss = self.trainer.train(model)
        self.lora = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if 'lora_B' in key
        }


class Server(FTBaseServer):
    def aggregate(self):
        total = sum(len(client.dataset['train']) for client in self.sampled_clients)
        if total <= 0:
            raise ValueError('FFA-LoRA requires sampled clients with training rows')
        b_keys = {key for key in self.global_lora if 'lora_B' in key}
        if not b_keys or any(set(client.lora) != b_keys for client in self.sampled_clients):
            raise ValueError('FFA-LoRA clients must upload exactly the B factors')
        updated = dict(self.global_lora)
        for key in b_keys:
            updated[key] = sum(
                client.lora[key] * (len(client.dataset['train']) / total)
                for client in self.sampled_clients
            )
        # The unchanged A is retained for client synchronization and final
        # evaluation; clients upload B only.
        self.global_lora = updated
        self.model.load_state_dict(self.global_lora, strict=False)
        print('FFA-LoRA aggregated B and retained the fixed A')
