"""Validate every fixed zsRE sample; optionally smoke-test GPU generation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import get_format_func
from utils.model_utils import load_model, load_tokenizer
from utils.zsre_utils import encode_zsre_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-smoke', action='store_true')
    args = parser.parse_args()
    config = SimpleNamespace(model='Qwen3-1.7B', device=0, task_type='CAUSAL_LM', dataset='zsre')
    tokenizer = load_tokenizer(config)
    formatter = get_format_func(config, tokenizer)
    count, max_prompt, max_target = 0, 0, 0
    smoke_rows = []
    for split in ('train', 'test'):
        for client in range(8):
            rows = [json.loads(line) for line in Path(f'dataset/zsre/{split}/{client}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
            assert len(rows) == 500
            if split == 'test':
                smoke_rows.append(rows[0])
            for row in rows:
                prompt = encode_zsre_prompt(tokenizer, row['input_ids'])
                result = formatter(row)
                target = tokenizer(str(row['label']), add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
                assert result['input_ids'][:len(prompt)] == prompt
                assert result['labels'][:len(prompt)] == [-100] * len(prompt)
                assert [x for x in result['labels'] if x != -100] == target
                assert result['input_ids'][len(prompt):len(prompt) + len(target)] == target
                assert sum(result['attention_mask']) == len(prompt) + len(target)
                assert len(result['input_ids']) == len(result['labels']) == len(result['attention_mask']) == 512
                assert len(target) <= 32, 'Answer plus EOS must fit generation budget'
                count += 1
                max_prompt, max_target = max(max_prompt, len(prompt)), max(max_target, len(target))
    from eval_zsre import containment_score, generate_and_score
    assert containment_score('female', 'male') == 0
    assert containment_score('Romeo', 'Rome') == 0
    assert containment_score('Harvard University extra words', 'Harvard University') == 1
    print(json.dumps({'validated_rows': count, 'max_prompt_tokens': max_prompt, 'max_target_tokens_with_eos': max_target}))
    if args.gpu_smoke:
        import torch
        assert torch.cuda.is_available()
        model = load_model(config)
        metrics, predictions = generate_and_score(model, tokenizer, smoke_rows, 8)
        print(json.dumps({'gpu_smoke': metrics, 'predictions': predictions}, ensure_ascii=False))
        assert metrics['eos_rate'] >= 75, 'Smoke generation unexpectedly fails to terminate'


if __name__ == '__main__':
    main()
