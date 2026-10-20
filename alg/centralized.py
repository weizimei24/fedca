"""Centralized pooled-data LoRA baseline.

This intentionally reuses the ordinary training and aggregation base classes.
With ``cn: 1`` the aggregation is the identity, making each configured round
one deterministic pass over the pooled training data while preserving the
same checkpoint/evaluation lifecycle as the federated runs.
"""

import json
from pathlib import Path

from alg.ftbase import FTBaseClient, FTBaseServer
from utils.time_utils import time_record


class Client(FTBaseClient):
    @time_record
    def run(self, model):
        super().run(model)


class Server(FTBaseServer):
    def __init__(self, args, clients):
        super().__init__(args, clients)
        self.zsre_test_rows = None
        if args.dataset == 'zsre' and not getattr(args, '_shared', False):
            from eval_zsre import read_jsonl

            root = Path('dataset/zsre/test')
            self.zsre_test_rows = [read_jsonl(root / f'{i}.jsonl') for i in range(8)]
            if any(len(rows) != 500 for rows in self.zsre_test_rows):
                raise ValueError('Centralized zsRE expects 8 test groups of 500 rows')

    def run(self):
        self.sample()
        self.local_run()
        self.aggregate()

    def test_all(self):
        if self.zsre_test_rows is None:
            return super().test_all()

        from eval_zsre import generate_and_score, generation_metadata

        # The pooled training set is one virtual data owner in this codebase.
        # The eight original groups are used only for stratified evaluation.
        self.save_round_adapter(self.round)
        output = Path(self.args.suffix) / 'round_evaluation' / 'zsre' / f'round_{self.round:03d}'
        predictions_dir = output / 'predictions'
        predictions_dir.mkdir(parents=True, exist_ok=True)
        per_group = {}
        for group_id, rows in enumerate(self.zsre_test_rows):
            metrics, predictions = generate_and_score(
                self.model, self.clients[0].tokenizer, rows, self.args.eval_batch_size
            )
            per_group[str(group_id)] = metrics
            with (predictions_dir / f'client_{group_id}.jsonl').open('w', encoding='utf-8') as handle:
                for row in predictions:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(
                f'Round {self.round} | test group {group_id} | '
                f'EM: {metrics["em"]:.2f} F1: {metrics["f1"]:.2f} '
                f'n: {metrics["num_samples"]}'
            )
        keys = ('em', 'f1', 'containment', 'num_samples',
                'average_generated_tokens', 'eos_rate', 'max_tokens_rate')
        macro = {key: sum(row[key] for row in per_group.values()) / 8 for key in keys}
        payload = {
            'method': 'centralized', 'round': self.round,
            'generation': generation_metadata(self.clients[0].tokenizer),
            'per_client': per_group, 'macro_average': macro,
        }
        (output / 'metrics.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
        )
        print(f'Round {self.round} | all 8 test groups | EM: {macro["em"]:.2f} F1: {macro["f1"]:.2f}')
        return {'em': macro['em'], 'f1': macro['f1'], 'containment': macro['containment']}
