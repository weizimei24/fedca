"""Shared Qwen3/SQuAD prompt protocol used by training and evaluation."""

from __future__ import annotations


SQUAD_MAX_LENGTH = 512
# The longest answer in the fixed 12k subset is 44 tokens. A fixed prompt
# budget keeps training and evaluation prompts identical and leaves room for
# the answer plus <|im_end|> during supervised training.
SQUAD_MAX_PROMPT_TOKENS = 448
SQUAD_MAX_NEW_TOKENS = 64
SQUAD_INSTRUCTION = (
    "Answer the question using only the provided context. "
    "Return only the short answer and no explanation."
)


def squad_user_content(context: str, question: str) -> str:
    return f"{SQUAD_INSTRUCTION}\nContext: {context}\nQuestion: {question}"


def _chat_prompt_ids(tokenizer, context: str, question: str) -> list[int]:
    messages = [{"role": "user", "content": squad_user_content(context, question)}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_squad_prompt(
    tokenizer,
    context: str,
    question: str,
    max_prompt_tokens: int = SQUAD_MAX_PROMPT_TOKENS,
) -> list[int]:
    """Encode a non-thinking Qwen3 prompt, truncating only its context."""
    prompt_ids = _chat_prompt_ids(tokenizer, context, question)
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt_ids

    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    lo, hi = 0, len(context_ids)
    best = _chat_prompt_ids(tokenizer, "", question)
    if len(best) > max_prompt_tokens:
        raise ValueError(
            f"Instruction and question require {len(best)} tokens, "
            f"exceeding prompt budget {max_prompt_tokens}"
        )

    # Find the largest context prefix whose complete chat prompt still fits.
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate_context = tokenizer.decode(context_ids[:mid], skip_special_tokens=True)
        candidate = _chat_prompt_ids(tokenizer, candidate_context, question)
        if len(candidate) <= max_prompt_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def encode_squad_training_example(tokenizer, example: dict) -> dict:
    prompt_ids = encode_squad_prompt(tokenizer, example["context"], example["question"])
    answer_ids = tokenizer(str(example["label"]), add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id for SQuAD training")

    available = SQUAD_MAX_LENGTH - len(prompt_ids) - 1
    if available <= 0:
        raise ValueError("SQuAD prompt leaves no room for an answer and EOS token")
    if len(answer_ids) > available:
        answer_ids = answer_ids[:available]

    target_ids = answer_ids + [eos_id]
    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    padding = SQUAD_MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding
    attention_mask += [0] * padding
    labels += [-100] * padding
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
