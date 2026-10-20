"""Shared factual QA protocol: non-thinking chat, short answer, supervised EOS."""

ZSRE_PROTOCOL = "zsre_chat_eos_v2"
ZSRE_MAX_LENGTH = 512
ZSRE_MAX_PROMPT_TOKENS = 448
ZSRE_MAX_NEW_TOKENS = 32
ZSRE_INSTRUCTION = "Answer the question. Return only the short answer and no explanation."


def encode_zsre_prompt(tokenizer, question):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{ZSRE_INSTRUCTION}\nQuestion: {question}"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    if len(ids) > ZSRE_MAX_PROMPT_TOKENS:
        raise ValueError("zsRE question exceeds prompt budget; refusing silent truncation")
    return ids


def encode_zsre_training_example(tokenizer, example):
    prompt = encode_zsre_prompt(tokenizer, example["input_ids"])
    answer = tokenizer(str(example["label"]), add_special_tokens=False)["input_ids"]
    if not answer or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("zsRE requires a nonempty answer and tokenizer EOS/pad IDs")
    target = answer + [tokenizer.eos_token_id]
    if len(prompt) + len(target) > ZSRE_MAX_LENGTH:
        raise ValueError("zsRE answer exceeds training budget; refusing silent truncation")
    padding = ZSRE_MAX_LENGTH - len(prompt) - len(target)
    return {
        "input_ids": prompt + target + [tokenizer.pad_token_id] * padding,
        "attention_mask": [1] * (len(prompt) + len(target)) + [0] * padding,
        "labels": [-100] * len(prompt) + target + [-100] * padding,
    }
