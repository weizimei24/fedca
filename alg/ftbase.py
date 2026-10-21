import os
import random

from peft import get_peft_model
from utils.data_utils import load_data, load_global_test_data
from utils.sys_utils import device_config
from utils.train_utils import Trainer
from utils.eval_utils import Evaluator
from alg.base import BaseClient, BaseServer
from utils.model_utils import load_model, load_tokenizer, load_lora_config
from utils.time_utils import time_record


class FTBaseClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.tokenizer = load_tokenizer(args)
        self.dataset = load_data(args=args, idx=self.id)
        self.lora = {}
        self.last_train_loss = None
        self.trainer = Trainer(args=args, dataset=self.dataset, client=self)
        self.evaluator = None if (args.global_test or getattr(args, 'cross_domain_qa', False) or
                                  getattr(args, 'commonsense', False) or getattr(args, '_shared', False)) else Evaluator(args=args, dataset=self.dataset)

    @time_record
    def run(self, model):
        self.last_train_loss = self.trainer.train(model)
        self.lora = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    def local_test(self, model):
        if self.evaluator is None:
            raise RuntimeError("Client-local evaluation is disabled for global_test or cross-domain QA runs")
        return self.evaluator.evaluate(model, round_idx=self.server.round, client_id=self.id)

class FTBaseServer(BaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.model = get_peft_model(load_model(args), load_lora_config(args))

        self.global_lora = {k: v.clone() for k, v in self.model.state_dict().items() if "lora_" in k}
        self.sample_rate = args.sr
        self.wall_clock_time = 0
        self.round = 0

        if getattr(args, '_shared', False):
            from utils.method_state import configure_trainable
            configure_trainable(self)
            self.global_test_dataset = None
            self.global_evaluator = None
        elif getattr(args, 'commonsense', False):
            self.global_test_dataset = None
            self.global_evaluator = None
        elif args.global_test:
            self.global_test_dataset = load_global_test_data(args)
            self.global_evaluator = Evaluator(args=args, dataset=self.global_test_dataset)
        elif getattr(args, 'cross_domain_qa', False):
            # This evaluator operates on raw named test files and performs
            # generation-based MRQA metrics independently for each domain.
            from utils.cross_domain_qa import CrossDomainQAEvaluator
            self.global_test_dataset = None
            self.global_evaluator = CrossDomainQAEvaluator(args)
        else:
            self.global_test_dataset = None
            self.global_evaluator = None

        for client, delay in zip(clients, device_config(args)): client.delay = delay

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def sample(self):
        sample_num = int(self.sample_rate * len(self.clients))
        self.sampled_clients = sorted(random.sample(self.clients, sample_num), key=lambda x: x.id)

    def local_run(self):
        continuation = getattr(self, 'experiment_run', None) or getattr(self, 'commonsense_run', None)
        for client in self.sampled_clients:
            if continuation is not None and continuation.restore_completed_client(client):
                continue
            client.run(self.model)
            if continuation is not None:
                continuation.commit_client(client)
            self.model.load_state_dict(self.global_lora, strict=False)
        self.wall_clock_time += max([c.training_time for c in self.sampled_clients])
        if getattr(self, 'experiment_run', None) is not None:
            self.experiment_run.before_aggregate()

    def aggregate(self):
        from utils.runtime_utils import factor_average
        self.global_lora = factor_average([client.lora for client in self.sampled_clients],
            [len(client.dataset['train']) for client in self.sampled_clients])
        self.model.load_state_dict(self.global_lora, strict=False)
        print("Aggregated model updated.")

    def export_state(self):
        from utils.method_state import export_state
        return export_state(self)

    def restore_state(self, state):
        from utils.method_state import restore_state
        restore_state(self, state)

    def test_all(self):
        if getattr(self, 'experiment_run', None) is not None:
            return self.experiment_run.evaluate(self.round + 1)
        if getattr(self.args, 'commonsense', False):
            return self.commonsense_run.evaluate(self.round + 1)
        if getattr(self.args, 'cross_domain_qa', False):
            print("Testing aggregated global model on named extractive-QA domain test sets ...")
            return self.global_evaluator.evaluate(
                self.model, self.clients[0].tokenizer, round_idx=self.round, method=self.args.alg
            )
        if self.global_evaluator is not None:
            print("Testing aggregated global model on shared global test set ...")
            metrics = self.global_evaluator.evaluate(
                self.model, round_idx=self.round, client_id=None
            )
            if self.args.round_generation_metrics:
                # Import lazily to keep the ordinary training path lightweight
                # and avoid loading generation evaluation code for other tasks.
                from eval import (
                    _ExactMatchEvaluator,
                    _F1Evaluator,
                    _generate_predictions,
                    _save_predictions,
                )

                predictions = _generate_predictions(
                    self.model, self.clients[0].tokenizer, self.args, client_idx=None
                )
                metrics.update(_ExactMatchEvaluator().evaluate(predictions))
                metrics.update(_F1Evaluator().evaluate(predictions))
                prediction_path = _save_predictions(
                    predictions,
                    self.args,
                    client_idx=None,
                    filename=f'global_round_{self.round}_predictions.jsonl',
                )
                print(f'Round {self.round} predictions saved to {prediction_path}')
            return metrics

        all_metrics = []
        for client in self.clients:
            print(f"Testing on client {client.id} ...")
            metrics = client.local_test(self.model)
            all_metrics.append(metrics)

        res_dict = {}
        for k in all_metrics[0].keys():
            res_dict[k] = sum(m[k] for m in all_metrics) / len(all_metrics)

        return res_dict

    def _adapter_path(self):
        args = self.args
        name = (
            f'{args.alg}_{args.dataset}_{args.model}_'
            f'{args.cn}c_{args.epoch}E_lr{args.lr}'
        )
        return os.path.join(args.suffix, 'adapter', name)

    def save_round_adapter(self, round_idx):
        """Persist the post-aggregation global adapter for audit/re-evaluation."""
        import torch
        adapter_path = os.path.join(self._adapter_path(), f'round_{round_idx:03d}')
        os.makedirs(adapter_path, exist_ok=True)
        torch.save(dict(self.global_lora), os.path.join(adapter_path, 'lora_weights.pt'))
        print(f'Round {round_idx} adapter saved to {adapter_path}')

    def save_client_updates(self, round_idx):
        """Persist each sampled client's post-training LoRA factors (Δ_i) for this round."""
        import torch
        for client in self.sampled_clients:
            client_dir = os.path.join(self._adapter_path(), f'round_{round_idx:03d}', 'clients')
            os.makedirs(client_dir, exist_ok=True)
            torch.save(dict(client.lora), os.path.join(client_dir, f'client_{client.id}.pt'))

    def save_adapter(self):
        import json
        import torch
        args = self.args
        adapter_path = self._adapter_path()
        os.makedirs(adapter_path, exist_ok=True)
        torch.save(dict(self.global_lora), os.path.join(adapter_path, 'lora_weights.pt'))
        with open(os.path.join(adapter_path, 'training_config.json'), 'w', encoding='utf-8') as handle:
            json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
        print(f'Adapter saved to {adapter_path}')
