from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record

class Client(FTBaseClient):
    @time_record
    def run(self, model):
        # LoRA starts with B=0, so the first (zero-indexed) round must update B.
        # Later rounds alternate B, A, B, A as in RoLoRA.
        active_factor = "lora_B" if self.server.round % 2 == 0 else "lora_A"
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if active_factor in name:
                param.requires_grad = True
        self.last_train_loss = self.trainer.train(model)
        self.lora = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if active_factor in key
        }

class Server(FTBaseServer):
    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def aggregate(self):
        # global_lora must retain BOTH factors: local_run() restores this full
        # checkpoint after every client, and save_adapter() uses it for final eval.
        active_factor = "lora_B" if self.round % 2 == 0 else "lora_A"
        active_keys = {key for key in self.global_lora if active_factor in key}
        if not active_keys or not self.sampled_clients:
            raise ValueError("RoLoRA requires active LoRA factors and sampled clients")
        if any(set(client.lora) != active_keys for client in self.sampled_clients):
            raise ValueError("RoLoRA client update does not match the active factor")
        total = sum(len(client.dataset['train']) for client in self.sampled_clients)
        if total <= 0:
            raise ValueError("RoLoRA sampled clients have no training rows")
        updated = dict(self.global_lora)
        for key in active_keys:
            updated[key] = sum(
                client.lora[key] * (len(client.dataset['train']) / total)
                for client in self.sampled_clients
            )
        self.global_lora = updated
        self.model.load_state_dict(self.global_lora, strict=False)
        print(f"RoLoRA aggregated {active_factor}; retained the frozen factor")
