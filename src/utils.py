"""
src/utils.py - Model loading, generation constants, seed management, and factories.
"""
import json
import os
from typing import Any, Callable

# Global random seed specified by course policy
GLOBAL_SEED = 5326

# Sampling & decoding hyperparameter configurations
GREEDY_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 512,
}

SAMPLING_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_new_tokens": 512,
}

JUDGE_GEN_KWARGS = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens": 96,
}


def set_seed_for_problem(problem_index: int) -> None:
    """
    Sets a deterministic seed for problem_index (5326 + problem_index).
    Guarantees reproducibility and safe resumability after interruptions.
    """
    try:
        import transformers
        transformers.set_seed(GLOBAL_SEED + problem_index)
    except ImportError:
        import random
        random.seed(GLOBAL_SEED + problem_index)


def load_models(generator_device: str = "cuda:0", judge_device: str = "cuda:1"):
    """
    Loads Qwen2.5-1.5B-Instruct in float16 (Generator) and
    Qwen2.5-7B-Instruct in 4-bit NF4 (Judge).
    
    Hardware Mapping:
      - Kaggle GPU T4x2: generator on cuda:0, judge on cuda:1.
      - Single GPU: both models placed on cuda:0.
      - Warning: Never use device_map="auto" across two GPUs for a single model!
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        g_dev = "cuda:0" if num_gpus == 1 else "cpu"
        j_dev = "cuda:0" if num_gpus == 1 else "cpu"
    else:
        g_dev = generator_device
        j_dev = judge_device

    # 1. Generator: Qwen/Qwen2.5-1.5B-Instruct in float16 (~3.1 GB VRAM)
    gen_id = "Qwen/Qwen2.5-1.5B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_id)
    if gen_tokenizer.pad_token_id is None:
        gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id

    gen_device_map = {"": g_dev} if g_dev != "cpu" else None
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_id,
        torch_dtype=torch.float16 if g_dev != "cpu" else torch.float32,
        device_map=gen_device_map,
    )
    if g_dev == "cpu":
        gen_model = gen_model.to("cpu")

    # 2. Judge: Qwen/Qwen2.5-7B-Instruct in 4-bit NF4 (~5.5 GB VRAM)
    judge_id = "Qwen/Qwen2.5-7B-Instruct"
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_id)
    if judge_tokenizer.pad_token_id is None:
        judge_tokenizer.pad_token_id = judge_tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    judge_device_map = {"": j_dev} if j_dev != "cpu" else None
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_id,
        quantization_config=bnb_config if j_dev != "cpu" else None,
        torch_dtype=torch.float16 if j_dev != "cpu" else torch.float32,
        device_map=judge_device_map,
    )
    if j_dev == "cpu":
        judge_model = judge_model.to("cpu")

    return gen_model, gen_tokenizer, judge_model, judge_tokenizer


def _get_eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Extracts all EOS token IDs from model.generation_config and tokenizer."""
    eos_ids: set[int] = set()
    if hasattr(model, "generation_config") and model.generation_config is not None:
        cfg_eos = getattr(model.generation_config, "eos_token_id", None)
        if cfg_eos is not None:
            if isinstance(cfg_eos, int):
                eos_ids.add(cfg_eos)
            elif isinstance(cfg_eos, (list, tuple, set)):
                eos_ids.update(cfg_eos)
    if tokenizer is not None:
        tok_eos = getattr(tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            if isinstance(tok_eos, int):
                eos_ids.add(tok_eos)
            elif isinstance(tok_eos, (list, tuple, set)):
                eos_ids.update(tok_eos)
    return eos_ids


def _slice_to_first_eos(gen_tokens: Any, eos_ids: set[int]) -> Any:
    """Slices token sequence up to and including the first EOS token."""
    if hasattr(gen_tokens, "tolist"):
        ids = gen_tokens.tolist()
    else:
        ids = list(gen_tokens)
    first_eos_idx = None
    for idx, tok in enumerate(ids):
        if tok in eos_ids:
            first_eos_idx = idx
            break
    if first_eos_idx is not None:
        return gen_tokens[:first_eos_idx + 1]
    return gen_tokens


def make_hf_generate_fn(model: Any, tokenizer: Any) -> Callable[[str, int, dict], list[dict]]:
    """
    Factory creating a GenerateFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw instruction prompts.
    """
    def generate_fn(prompt: str, num_samples: int, gen_kwargs: dict) -> list[dict]:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(gen_kwargs)
        if not kwargs.get("do_sample", False):
            kwargs["temperature"] = None
            kwargs["top_p"] = None
            kwargs["top_k"] = None

        if num_samples > 1 and kwargs.get("do_sample", True):
            kwargs["num_return_sequences"] = num_samples

        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(model, tokenizer)
        results = []
        for i in range(output_ids.shape[0]):
            gen_tokens = output_ids[i][prompt_len:]
            gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
            text = tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
            results.append({"text": text, "num_tokens": len(gen_tokens_valid)})
        return results

    return generate_fn


def make_hf_judge_fn(judge_model: Any, judge_tokenizer: Any) -> Callable[[str], dict]:
    """
    Factory creating a JudgeFn conforming to submission_adapter contract.
    Applies the Qwen chat template to raw judge prompts.
    """
    def judge_fn(judge_prompt: str) -> dict:
        try:
            import torch
            no_grad_ctx = torch.no_grad
        except ImportError:
            class _NoGrad:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            no_grad_ctx = lambda: _NoGrad()

        messages = [{"role": "user", "content": judge_prompt}]
        formatted = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = judge_tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(judge_model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(JUDGE_GEN_KWARGS)
        kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)

        with no_grad_ctx():
            output_ids = judge_model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)
        gen_tokens = output_ids[0][prompt_len:]
        gen_tokens_valid = _slice_to_first_eos(gen_tokens, eos_ids)
        text = judge_tokenizer.decode(gen_tokens_valid, skip_special_tokens=True)
        return {"text": text, "num_tokens": len(gen_tokens_valid)}

    return judge_fn


def make_hf_batch_judge_fn(judge_model: Any, judge_tokenizer: Any) -> Callable[[list[str]], list[dict]]:
    """
    Batched variant of make_hf_judge_fn: judges many prompts in ONE generate() call.
    Same chat template, same JUDGE_GEN_KWARGS (greedy, max_new_tokens=96), same
    first-EOS slicing and token counting as the single-prompt version.

    Prompts are LEFT-padded so every sequence ends right where generation starts
    (required for batched generation with decoder-only models).
    Returns one {"text": str, "num_tokens": int} per prompt, in input order.
    """
    import torch

    def batch_judge_fn(prompts: list[str]) -> list[dict]:
        if not prompts:
            return []
        texts = [
            judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        old_side = judge_tokenizer.padding_side
        judge_tokenizer.padding_side = "left"
        try:
            inputs = judge_tokenizer(texts, return_tensors="pt", padding=True)
        finally:
            judge_tokenizer.padding_side = old_side
        inputs = {k: v.to(judge_model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        kwargs = dict(JUDGE_GEN_KWARGS)
        kwargs.setdefault("pad_token_id", judge_tokenizer.eos_token_id)

        with torch.no_grad():
            output_ids = judge_model.generate(**inputs, **kwargs)

        eos_ids = _get_eos_token_ids(judge_model, judge_tokenizer)
        results = []
        for i in range(output_ids.shape[0]):
            gen_tokens = _slice_to_first_eos(output_ids[i][prompt_len:], eos_ids)
            text = judge_tokenizer.decode(gen_tokens, skip_special_tokens=True)
            results.append({"text": text, "num_tokens": len(gen_tokens)})
        return results

    return batch_judge_fn


def make_hf_verdict_logprob_fn(
    judge_model: Any,
    judge_tokenizer: Any,
    verdict_prefix: str = "Verdict:",
    valid_text: str = " VALID",
    invalid_text: str = " INVALID",
) -> Callable[[list[str]], list[dict]]:
    """
    (Optional bonus) Continuous step confidence from the judge's raw logits.

    For each judge prompt, the context is the chat-templated prompt followed by the
    tokens of "Verdict:" -- exactly the context in which the judge wrote its verdict
    during the main run (every judge output starts with "Verdict: VALID|INVALID").
    We then compute the FULL-SEQUENCE log-probability of each continuation
    (" VALID" and " INVALID"), i.e. the sum of log-probs of all of its tokens with
    teacher forcing, so the method is correct even if a word splits into several
    tokens. The first-token comparison is returned too.

    Returns one dict per prompt:
      logp_valid, logp_invalid            full-sequence log-probs
      first_logp_valid, first_logp_invalid log-probs of the FIRST token of each word
      n_tokens_valid, n_tokens_invalid     token lengths of the two continuations
    """
    import torch

    valid_ids = judge_tokenizer.encode(valid_text, add_special_tokens=False)
    invalid_ids = judge_tokenizer.encode(invalid_text, add_special_tokens=False)
    prefix_tail = judge_tokenizer.encode(verdict_prefix, add_special_tokens=False)
    alternatives = [valid_ids, invalid_ids]
    keep = max(len(a) for a in alternatives)  # logits needed at the last `keep` positions
    pad_id = judge_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = judge_tokenizer.eos_token_id

    def logprob_fn(prompts: list[str]) -> list[dict]:
        if not prompts:
            return []
        # One context sequence per (prompt, alternative): prefix + all but the last token
        # of the alternative. Identical sequences are evaluated once (with single-token
        # words both alternatives share the same context -> one forward pass per prompt).
        seqs: list[list[int]] = []
        seq_index: dict[tuple, int] = {}
        plan = []  # per prompt: [(seq_idx, alt_ids) for each alternative]
        for p in prompts:
            chat = judge_tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            prefix = judge_tokenizer(chat, add_special_tokens=False)["input_ids"] + prefix_tail
            row = []
            for alt in alternatives:
                ctx = tuple(prefix + alt[:-1])
                if ctx not in seq_index:
                    seq_index[ctx] = len(seqs)
                    seqs.append(list(ctx))
                row.append((seq_index[ctx], alt))
            plan.append(row)

        # Left-pad so every context ends at the last position.
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, width - len(s):] = torch.tensor(s, dtype=torch.long)
            attention[i, width - len(s):] = 1
        position_ids = (attention.cumsum(-1) - 1).clamp(min=0)
        device = judge_model.device
        input_ids, attention, position_ids = input_ids.to(device), attention.to(device), position_ids.to(device)

        with torch.no_grad():
            try:
                out = judge_model(input_ids=input_ids, attention_mask=attention,
                                  position_ids=position_ids, logits_to_keep=keep)
            except TypeError:  # older transformers without logits_to_keep
                out = judge_model(input_ids=input_ids, attention_mask=attention, position_ids=position_ids)
            logprobs = torch.log_softmax(out.logits[:, -keep:, :].float(), dim=-1).cpu()

        results = []
        for row in plan:
            full, first = [], []
            for seq_i, alt in row:
                m = len(alt)
                # token j of the alternative is predicted at position -(m - j) of its context
                lp = [logprobs[seq_i, keep - (m - j), tok].item() for j, tok in enumerate(alt)]
                full.append(sum(lp))
                first.append(lp[0])
            results.append({
                "logp_valid": full[0], "logp_invalid": full[1],
                "first_logp_valid": first[0], "first_logp_invalid": first[1],
                "n_tokens_valid": len(valid_ids), "n_tokens_invalid": len(invalid_ids),
            })
        return results

    logprob_fn.num_sequences_per_prompt = len({tuple(a[:-1]) for a in alternatives})  # type: ignore
    return logprob_fn


def save_results_summary(results_list: list[dict[str, Any]], filepath: str = "outputs/results_summary.json") -> None:
    """
    Saves results_summary.json conforming strictly to the assignment schema:
    Keys: 'Method', 'Accuracy (%)', 'Generation Calls', 'Judge Calls', 'Total LLM Calls',
          'Avg Gen Tokens / Problem', 'Avg Judge Tokens / Problem', 'Total Time (s)'
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2)


def save_aggregation_ablation(ablation_list: list[dict[str, Any]], filepath: str = "outputs/aggregation_ablation.json") -> None:
    """
    Saves aggregation_ablation.json:
    List of {'Strategy': str, 'Lambda': float | None, 'Accuracy (%)': float, 'Tie Rate (%)': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(ablation_list, f, indent=2)


def save_sc_scaling_curve(curve_list: list[dict[str, Any]], filepath: str = "outputs/sc_scaling_curve.json") -> None:
    """
    Saves sc_scaling_curve.json:
    List of {'N': int, 'Accuracy (%)': float, 'Avg Gen Tokens / Problem': float}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(curve_list, f, indent=2)
