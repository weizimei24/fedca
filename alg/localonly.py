"""Independent domain-local LoRA models with no server aggregation."""

from __future__ import annotations

import json
import os

import torch

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)


class Server(FTBaseServer):
    """Keep one persistent LoRA state per client and never aggregate them."""

    def __init__(self, args, clients):
        super().__init__(args, clients)
        if float(args.sr) != 1.0:
            raise ValueError("localonly requires sr: 1.0 so every client trains each round")
        self.cross_domain_qa = bool(getattr(args, "cross_domain_qa", False))
        if self.cross_domain_qa:
            if self.global_evaluator is None:
                raise ValueError("localonly cross-domain QA requires its named-domain evaluator")
            self.client_domains = self.global_evaluator.client_domains
            if len(clients) != len(self.client_domains):
                raise ValueError(
                    f"localonly cross-domain QA requires {len(self.client_domains)} clients, got {len(clients)}"
                )
        elif getattr(args, '_shared', False):
            self.client_domains = tuple(str(c.id) for c in clients)
        elif getattr(args, 'commonsense', False):
            from utils.commonsense_training import DOMAINS
            if len(clients) != len(DOMAINS):
                raise ValueError('Commonsense localonly requires eight source clients')
            self.client_domains = DOMAINS
        elif args.dataset != "zsre":
            raise ValueError("localonly currently supports cross_domain_qa or dataset: zsre")
        self.client_loras = {
            client.id: {key: value.clone() for key, value in self.global_lora.items()}
            for client in clients
        }

    def run(self):
        self.sample()
        self.local_run()
        # Intentionally no call to aggregate(): local adapters remain separate.

    def local_run(self):
        continuation = getattr(self, 'experiment_run', None) or getattr(self, 'commonsense_run', None)
        for client in self.sampled_clients:
            if continuation is not None and continuation.restore_completed_client(client):
                continue
            self.model.load_state_dict(self.client_loras[client.id], strict=False)
            client.run(self.model)
            self.client_loras[client.id] = {
                key: value.clone() for key, value in client.lora.items()
            }
            if continuation is not None:
                continuation.commit_client(client)
        # Leave the shared model container in its neutral initial state. Every
        # subsequent train/eval explicitly loads the intended client's LoRA.
        self.model.load_state_dict(self.global_lora, strict=False)
        self.wall_clock_time += max(client.training_time for client in self.sampled_clients)
        if getattr(self, 'experiment_run', None) is not None:
            # All clients may have been restored without running Trainer.
            # Discard the replayed sample() RNG draw, as federated methods do.
            self.experiment_run.before_aggregate()

    def aggregate(self):
        """No-op required by the common server interface."""

    def get_round_train_metrics(self) -> dict[str, float]:
        metrics = {}
        for client in self.sampled_clients:
            if client.last_train_loss is not None:
                if self.cross_domain_qa:
                    domain = self.client_domains[client.id]
                    label = f"client_{client.id}_{domain}" if self.global_evaluator.has_repeated_domains else domain
                else:
                    label = f"client_{client.id}"
                metrics[f"{label}_train_loss"] = client.last_train_loss
        return metrics

    def test_all(self):
        if getattr(self, 'experiment_run', None) is not None:
            return self.experiment_run.evaluate(self.round + 1)
        if getattr(self.args, 'commonsense', False):
            return self.commonsense_run.evaluate(self.round + 1)
        if not self.cross_domain_qa:
            all_metrics = []
            for client in self.clients:
                self.model.load_state_dict(self.client_loras[client.id], strict=False)
                all_metrics.append(client.local_test(self.model))
            self.model.load_state_dict(self.global_lora, strict=False)
            return {
                key: sum(metrics[key] for metrics in all_metrics) / len(all_metrics)
                for key in all_metrics[0]
            }

        per_domain = []
        for client in self.clients:
            domain = self.client_domains[client.id]
            self.model.load_state_dict(self.client_loras[client.id], strict=False)
            per_domain.append(
                self.global_evaluator.evaluate_domain(
                    self.model,
                    client.tokenizer,
                    domain=domain,
                    round_idx=self.round,
                    method=self.args.alg,
                    client_id=client.id if self.global_evaluator.has_repeated_domains else None,
                )
            )
        self.model.load_state_dict(self.global_lora, strict=False)
        macro = self.global_evaluator.append_macro(per_domain, self.round, self.args.alg)

        metrics = {}
        for row in per_domain:
            domain = row["domain"]
            label = f"client_{row['client_id']}_{domain}" if self.global_evaluator.has_repeated_domains else domain
            metrics[f"{label}_em"] = row["em"]
            metrics[f"{label}_f1"] = row["f1"]
            metrics[f"{label}_containment"] = row["containment"]
        metrics.update({
            "macro_em": macro["em"],
            "macro_f1": macro["f1"],
            "macro_containment": macro["containment"],
        })
        return metrics

    def save_round_adapter(self, round_idx):
        """No global adapter exists; per-client snapshots are saved separately."""
        print(f"Round {round_idx}: local-only has no global adapter to save")

    def save_adapter(self):
        adapter_path = self._adapter_path()
        os.makedirs(adapter_path, exist_ok=True)
        for client_id, lora in self.client_loras.items():
            torch.save(dict(lora), os.path.join(adapter_path, f"client_{client_id}.pt"))
        with open(os.path.join(adapter_path, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(vars(self.args), handle, ensure_ascii=False, indent=2, default=str)
        print(f"Final local adapters saved to {adapter_path}")
