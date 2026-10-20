import os
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType


def local_model_fingerprint(directory):
    """Validate local shard indexes and fingerprint model/tokenizer assets."""
    from utils.runtime_utils import sha256
    root = Path(directory).resolve()
    if not (root / 'config.json').is_file():
        raise ValueError(f'Local model config.json is missing: {root}')
    files = set(root.glob('*.safetensors')) | set(root.glob('pytorch_model*.bin'))
    for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
        index = root / name
        if not index.exists():
            continue
        mapping = json.loads(index.read_text(encoding='utf-8')).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f'Invalid model shard index: {index}')
        for shard in mapping.values():
            if not isinstance(shard, str):
                raise ValueError(f'Invalid model shard name in {index}')
            relative = Path(shard)
            target = root / relative
            # HF cache snapshots legitimately symlink shards to ../blobs.
            # Reject traversal in the INDEX, while allowing such local links.
            if relative.is_absolute() or '..' in relative.parts or not target.is_file():
                raise ValueError(f'Local model shard is missing or outside model directory: {shard}')
            files.add(target)
    if not files:
        raise ValueError(f'Local model weights are missing: {root}')
    # Includes indexes, generation defaults, added tokens and chat templates;
    # a tokenizer change must not silently reuse an old optimizer checkpoint.
    for pattern in ('*.json', '*.model', '*.tiktoken', '*.vocab', '*.merges',
                    'vocab.txt', 'merges.txt', '*.jinja', 'chat_templates/*.jinja'):
        files.update(root.glob(pattern))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}

def load_model(args):
    # Honour the configured GPU instead of unconditionally selecting GPU 3.
    # This must happen before the first CUDA model allocation.
    requested_device = str(args.device)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_device
    elif visible_devices != requested_device:
        print(
            f"CUDA_VISIBLE_DEVICES={visible_devices} is already set; "
            f"leaving it in place instead of overriding --device {requested_device}."
        )
    model_name = args.model
    
    if args.task_type == 'SEQ_CLS':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            device_map="auto",
            # torch_dtype=torch.float16
        )
    elif args.task_type == 'CAUSAL_LM':
        if getattr(args, 'commonsense', False) or getattr(args, '_shared', False):
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float16,
                local_files_only=True,
            )
            model.config.use_cache = False
            if getattr(args, 'gradient_checkpointing', False):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            return model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # auto / cuda:0 / cpu ...
            torch_dtype=torch.float16
        )
    return model

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return tokenizer

def load_lora_config(args):
    if args.task_type == 'SEQ_CLS':
        lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],  # BERT applies "query", "value" instead of "q_proj"/"v_proj"
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    elif args.task_type == 'CAUSAL_LM':
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
    return lora_config
