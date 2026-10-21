import importlib.util
import sys
import numpy as np
import os
from pathlib import Path

from utils.logger import get_logger
from utils.options import args_parser
from utils.seed_utils import set_global_seed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FedSim:
    def __init__(self, args):
        from utils.experiment import use_shared_runtime
        args._shared = use_shared_runtime(args)
        if args._shared and args.suffix == 'default':
            dataset_name = Path(args.dataset).name
            args.suffix = f'shared/{args.task or dataset_name}/{dataset_name}/{args.alg}'
        if args._shared or getattr(args, 'commonsense', False):
            from utils.commonsense_training import run_lock
            suffix = Path(args.suffix)
            if suffix.is_absolute() or '..' in suffix.parts:
                raise ValueError('Experiment suffix must be a relative path below exp/')
            with run_lock(Path('exp') / suffix):
                self._initialize(args)
        else:
            self._initialize(args)

    def _initialize(self, args):
        set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
        print(f"Reproducibility: seed={args.seed}, deterministic={args.deterministic}")
        self.args = args
        args.suffix = f'exp/{args.suffix}'
        if args._shared:
            from utils.experiment import prepare_experiment
            prepare_experiment(args)
        elif getattr(args, 'commonsense', False):
            from utils.commonsense_framework import prepare_framework
            prepare_framework(args)
        self.logger = get_logger(args)

        # === route to algorithm module ===
        if args.mode == 'prototype':
            alg_module = importlib.import_module(f'alg.{args.alg}')
        elif args.mode == 'realistic':
            alg_module = importlib.import_module(f'alg.{args.alg}_event')

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.cn), desc="Loading clients...")]
        if args._shared and args.check_only:
            from utils.runtime_utils import atomic_json
            path = Path(args.suffix) / 'readiness.json'
            atomic_json(path, args._experiment_prepared['report'])
            print(f'Shared framework readiness passed: {path}; no model loaded or training started')
            return
        if getattr(args, 'commonsense', False) and args.check_only:
            from utils.commonsense_training import atomic_json
            path = Path('results/data_audits') / f'commonsense_framework_{args.alg}_readiness.json'
            atomic_json(path, args._commonsense_prepared['report'])
            print(f'Framework clients/Trainer/data readiness passed: {path}; no model loaded or training started')
            return
        # Client data loading must not influence the subsequent LoRA
        # initialization RNG state.
        set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
        self.server = alg_module.Server(args, self.clients)
        if args._shared:
            from utils.experiment import ExperimentRun
            self.server.experiment_run = ExperimentRun(args, self.server)
        elif getattr(args, 'commonsense', False):
            from utils.commonsense_framework import CommonsenseRun
            if args.alg == 'fedca':
                from utils.commonsense_fedca import CommonsenseFedCARun
                self.server.commonsense_run = CommonsenseFedCARun(args, self.server)
            else:
                self.server.commonsense_run = CommonsenseRun(args, self.server)
        
        if args.mode == 'prototype':
            self.simulate()
        elif args.mode == 'realistic':
            self.simulate_ddl()

    def simulate(self):
        TEST_GAP = self.args.test_gap
        continuation = getattr(self.server, 'experiment_run', None) or getattr(self.server, 'commonsense_run', None)
        if continuation is not None:
            continuation.recover_evaluations(eval_only=self.args.eval_only)
            if self.args.eval_only:
                return
        first_round = continuation.completed_rounds if continuation is not None else 0

        try:
            for rnd in tqdm(range(first_round, self.args.rnd), desc='Communication round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()
                if continuation is not None:
                    continuation.commit_round()

                # A one-client run is the repository's centralized baseline
                # (both ``alg: centralized`` and the legacy SQuAD ``cn: 1``
                # configuration). Record its round-average training loss even
                # when evaluation is skipped by ``test_gap``.
                if hasattr(self.server, 'get_round_train_metrics'):
                    train_metrics = self.server.get_round_train_metrics()
                elif self.args.cn == 1:
                    train_loss = getattr(self.server.sampled_clients[0], 'last_train_loss', None)
                    train_metrics = {} if train_loss is None else {'train_loss': train_loss}
                else:
                    train_metrics = {}
                if train_metrics:
                    train_metrics_str = '  '.join(
                        f'{key}: {value:.4f}' for key, value in train_metrics.items()
                    )
                    self.logger.info(
                        f'[Round {rnd}] {train_metrics_str}  '
                        f'wall_clock: {self.server.wall_clock_time}'
                    )

                # Post-round state is captured before evaluation. Federated
                # methods save client states plus a global adapter; local-only
                # saves only its four independent client states.
                if getattr(self.args, 'cross_domain_qa', False) and hasattr(self.server, 'save_round_adapter'):
                    self.server.save_client_updates(rnd)   # 新增：保存 4 个 Δ_i
                    self.server.save_round_adapter(rnd)

                # ===================== test =====================
                should_test = ((rnd+1) % TEST_GAP == 0 or rnd+1 == self.args.rnd) if continuation is not None else (
                    (self.args.rnd - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)))
                if should_test:
                    ret_dict = self.server.test_all()
                    metrics_str = '  '.join(f'{k}: {v:.4f}' for k, v in ret_dict.items())
                    self.logger.info(f'[Round {rnd}] {metrics_str}  wall_clock: {self.server.wall_clock_time}')

        except KeyboardInterrupt:
            if continuation is not None:
                print('Interrupted; rerun the same main.py command to resume the last committed checkpoint.', flush=True)
                raise
            ...
        finally:
            if continuation is None:
                self.server.save_adapter()
        if continuation is not None:
            continuation.mark_complete()


    def simulate_ddl(self):
        TEST_GAP = self.args.test_gap
        try:
            while self.server.early_break is False:
                self.server.run()

                if (self.args.rnd - self.server.round <= 10) or (self.server.round % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    metrics_str = '  '.join(f'{k}: {v:.4f}' for k, v in ret_dict.items())
                    self.logger.info(f'[Round {self.server.round}] {metrics_str}  wall_clock: {self.server.wall_clock_time}')

        except KeyboardInterrupt:
            ...
        finally:
            self.server.save_adapter()

if __name__ == '__main__':
    try:
        FedSim(args=args_parser())
    except KeyboardInterrupt:
        raise SystemExit(130)
