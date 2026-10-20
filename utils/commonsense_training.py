"""Compatibility imports for pre-shared Commonsense experiments.

New datasets use runtime_utils and dataset_adapters; this module keeps old
public imports stable without maintaining another checkpoint implementation.
"""
from utils.runtime_utils import *  # noqa: F401,F403


def encode_example(tokenizer, row, max_length):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["input_ids"]}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)
    target = tokenizer(row["label"], add_special_tokens=False)["input_ids"]
    if not target or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("A nonempty label and tokenizer EOS/pad tokens are required")
    target = target + [tokenizer.eos_token_id]
    if len(prompt) + len(target) > max_length:
        raise ValueError(f"{row['sample_id']} exceeds --max-length {max_length}; no silent truncation")
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}


def train_client(model, rows, client_id, round_index, cfg, active, on_checkpoint):
    """Compatibility diagnostic helper; optimization lives in the real Trainer."""
    from types import SimpleNamespace
    from utils.train_utils import Trainer

    class Continuation:
        def active_for(self, client):
            return active

        def checkpoint_active(self, client, state):
            on_checkpoint(state)

    args = SimpleNamespace(commonsense=True, task_type='CAUSAL_LM', seed=cfg['seed'],
                           bs=cfg['bs'], grad_accum=cfg['grad_accum'], lr=cfg['lr'],
                           epoch=1, step=0, checkpoint_steps=cfg['checkpoint_steps'])
    client = SimpleNamespace(id=client_id, tokenizer=SimpleNamespace(pad_token_id=cfg['pad_id']),
                             server=SimpleNamespace(round=round_index, commonsense_run=Continuation()))
    trainer = Trainer(args, {'train': rows}, client)
    trainer.train(model)
    return trainer.last_train_metrics
