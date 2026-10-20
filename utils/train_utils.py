import time
from functools import partial
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from utils.seed_utils import client_round_seed, seed_rngs


class Trainer:
    """
    Stateless trainer decoupled from model architecture.
    Instantiated once per client and reused across rounds.
    """

    def __init__(self, args, dataset, client):
        self.args = args
        self.client = client
        self.task_type = args.task_type
        self.shuffle_generator = torch.Generator()
        self.dynamic_batches = bool(getattr(args, 'commonsense', False) or getattr(args, '_shared', False))
        self.commonsense = self.dynamic_batches  # compatibility for existing Trainer subclasses
        collate_fn = None
        if self.dynamic_batches:
            from utils.runtime_utils import collate
            collate_fn = partial(collate, pad_id=client.tokenizer.pad_token_id, device=torch.device('cpu'))
        self.train_loader = DataLoader(
            dataset['train'],
            batch_size=self.args.bs,
            shuffle=True,
            drop_last=not self.dynamic_batches,
            generator=self.shuffle_generator,
            collate_fn=collate_fn,
        )

    def _set_round_seed(self):
        seed = client_round_seed(
            self.args.seed, self.client.id, self.client.server.round
        )
        seed_rngs(seed)
        self.shuffle_generator.manual_seed(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, model):
        """Run one round and return the mean unscaled loss over seen batches."""
        self._set_round_seed()
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler('cuda', enabled=next(model.parameters()).device.type == 'cuda') if self.dynamic_batches else GradScaler()

        train_loss = self._train_loop(model, optimizer, scaler)
        self._log(train_loss)
        return train_loss

    def train_windowed(self, model, start_time, window_end_time):
        """Run local training but stop early when the simulated window closes.

        Args:
            model: the model to train.
            start_time: simulated wall-clock datetime when training is assigned.
            window_end_time: simulated datetime at which the device's window
                closes; training stops after the first gradient update whose
                real elapsed time exceeds this simulated budget.

        Returns:
            (loss, truncated): the last computed loss and a bool indicating
            whether the run was cut short by the window deadline.
        """
        self._set_round_seed()
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        delay = getattr(self.client, 'delay', 1.0)
        loss, truncated, global_step = self._train_loop_windowed(model, optimizer, scaler, start_time, window_end_time, delay=delay)
        self._log(loss)
        return loss, truncated, global_step

    def train_resumed(self, model, start_step: int, start_time, window_end_time):
        """Resume windowed training from *start_step* completed gradient steps.

        Loads model weights from the caller (already restored externally) and
        continues for the remaining ``args.step - start_step`` optimizer steps,
        still subject to the window deadline.

        Returns:
            (loss, truncated, completed_steps)
        """
        self._set_round_seed()
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        delay = getattr(self.client, 'delay', 1.0)
        loss, truncated, global_step = self._train_loop_windowed(
            model, optimizer, scaler, start_time, window_end_time, resume_step=start_step, delay=delay
        )
        self._log(loss)
        return loss, truncated, global_step

    # ------------------------------------------------------------------
    # Per-task implementations
    # ------------------------------------------------------------------

    def _forward(self, model, batch):
        if self.task_type == 'SEQ_CLS':
            return self._forward_seq_cls(model, batch)
        elif self.task_type == 'CAUSAL_LM':
            return self._forward_causal_lm(model, batch)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _forward_seq_cls(self, model, batch):
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
        labels = torch.tensor(batch['labels']).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def _forward_causal_lm(self, model, batch):
        if self.dynamic_batches:
            device = next(model.parameters()).device
            tensors = {key: value.to(device) for key, value in batch.items()}
            with autocast(device.type, enabled=device.type == 'cuda'):
                return model(**tensors, use_cache=False).loss
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
        labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_loop(self, model, optimizer, scaler):
        if self.dynamic_batches:
            return self._train_loop_checkpointed(model, optimizer, scaler)
        accumulation_steps = self.args.grad_accum
        global_step = 0
        loss_sum = None
        batch_count = 0
        done = False

        optimizer.zero_grad()
        for _ in range(self.args.epoch):
            if done:
                break
            for step, batch in enumerate(self.train_loader):
                unscaled_loss = self._forward(model, batch)
                detached_loss = unscaled_loss.detach().float()
                loss_sum = detached_loss if loss_sum is None else loss_sum + detached_loss
                batch_count += 1
                loss = unscaled_loss / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step >= self.args.step:
                    done = True
                    break

        if batch_count == 0:
            raise RuntimeError("Training DataLoader produced no batches")
        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Synchronise with the device only once per round, rather than calling
        # ``.item()`` for every micro-batch in the hot training loop.
        return float((loss_sum / batch_count).item())

    def _train_loop_checkpointed(self, model, optimizer, scaler):
        """Original Trainer/DataLoader with optimizer-boundary continuation.

        This path adds dynamic collation and durable state for shared experiments;
        other datasets retain their existing training and stopping semantics.
        """
        from utils.runtime_utils import adapter_state, capture_rng, cpu_tree, load_adapter, restore_rng
        continuation = (getattr(self.client.server, 'experiment_run', None)
                        or self.client.server.commonsense_run)
        active = continuation.active_for(self.client)
        start_epoch, start_batch, updates = 0, 0, 0
        loss_sum, count, examples = 0.0, 0, 0
        if active is not None:
            load_adapter(model, active['adapter'])
            optimizer.load_state_dict(active['optimizer'])
            scaler.load_state_dict(active['scaler'])
            start_epoch, start_batch = active['epoch'], active['next_batch']
            updates, loss_sum = active['updates'], active['loss_sum']
            count, examples = active['batch_count'], active['examples']
            print(f"Resume round {self.client.server.round+1} client {self.client.id}: "
                  f"epoch {start_epoch}, batch {start_batch}, optimizer step {updates}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.args.epoch):
            iterator = iter(self.train_loader)
            if epoch < start_epoch:
                # Recreate earlier private-generator permutations; no forward or
                # optimizer step is replayed and global dropout RNG is untouched.
                # Do not ask the RandomSampler for StopIteration: its final
                # empty remainder can consume another private randperm. Match
                # the exact number of next() calls in the training loop.
                for _ in range(len(self.train_loader)):
                    next(iterator)
                continue
            cursor = start_batch if epoch == start_epoch else 0
            for _ in range(cursor):
                next(iterator)
            if active is not None and epoch == start_epoch:
                restore_rng(active['rng'])
            while cursor < len(self.train_loader) and (self.args.step <= 0 or updates < self.args.step):
                window = min(self.args.grad_accum, len(self.train_loader) - cursor)
                for _ in range(window):
                    batch = next(iterator)
                    loss = self._forward(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    loss_sum += loss.detach().float().item()
                    count += 1
                    examples += batch['input_ids'].shape[0]
                    scaler.scale(loss / window).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                cursor += window
                updates += 1
                finished = cursor == len(self.train_loader) or (self.args.step > 0 and updates == self.args.step)
                if updates % self.args.checkpoint_steps == 0 or finished:
                    checkpoint = {'client_id': self.client.id, 'round_index': self.client.server.round,
                                  'epoch': epoch, 'next_batch': cursor, 'updates': updates,
                                  'loss_sum': loss_sum, 'batch_count': count, 'examples': examples,
                                  'adapter': adapter_state(model), 'optimizer': cpu_tree(optimizer.state_dict()),
                                  'scaler': scaler.state_dict(), 'rng': capture_rng()}
                    continuation.checkpoint_active(self.client, checkpoint)
                    print(f"Round {self.client.server.round+1} client {self.client.id}: "
                          f"{cursor}/{len(self.train_loader)} batches; checkpoint saved", flush=True)
            if self.args.step > 0 and updates >= self.args.step:
                break
        if count == 0:
            raise RuntimeError('Training DataLoader produced no batches')
        self.last_train_metrics = {'client_id': self.client.id, 'mean_loss': loss_sum/count,
                                   'batches': count, 'optimizer_steps': updates, 'examples': examples,
                                   'lr': self.args.lr * 0.99 ** self.client.server.round}
        return loss_sum / count

    def _train_loop_windowed(self, model, optimizer, scaler, start_time, window_end_time, resume_step: int = 0, delay: float = 1.0):
        accumulation_steps = self.args.grad_accum
        global_step = resume_step
        loss = None
        done = False
        truncated = False

        # real_elapsed = sim_budget / delay  (because training_time = real_elapsed * delay)
        sim_budget = (window_end_time - start_time).total_seconds()
        real_deadline = time.time() + sim_budget / delay

        optimizer.zero_grad()
        for _ in range(self.args.epoch):
            if done:
                break
            for step, batch in enumerate(self.train_loader):
                loss = self._forward(model, batch) / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if time.time() > real_deadline:
                        
                        truncated = True
                        done = True
                        break

                if global_step >= self.args.step:
                    done = True
                    break

        if not truncated and (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return loss, truncated, global_step

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, loss):
        # Ordinary training returns a float containing the round-average
        # unscaled loss. Windowed training still supplies its last scaled
        # tensor, so retain backward-compatible handling for that path.
        if torch.is_tensor(loss):
            loss_value = loss.item() * self.args.grad_accum
        else:
            loss_value = float(loss)
        print(
            f"Round {self.client.server.round} | "
            f"Client {self.client.id} | "
            f"Train Loss: {loss_value:.4f}"
        )
