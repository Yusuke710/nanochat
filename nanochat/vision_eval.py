"""
OCR evaluation functions for vision benchmarks.

Fox: Precision (decoded text vs ground truth)
OmniDocBench: Normalized Edit Distance (primary metric)
"""

from collections import Counter

# -----------------------------------------------------------------------------
# Metrics

def edit_distance(s1, s2):
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def normalized_edit_distance(pred, gt):
    """Normalized edit distance (0 = perfect, 1 = completely wrong).
    Primary metric for OmniDocBench.
    """
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / max(len(pred), len(gt))


def precision(pred, gt):
    """Character-level precision. Used by Fox benchmark.
    Measures what fraction of predicted characters are correct.
    """
    if len(pred) == 0:
        return 1.0 if len(gt) == 0 else 0.0
    pred_counter = Counter(pred)
    gt_counter = Counter(gt)
    correct = sum((pred_counter & gt_counter).values())
    return correct / len(pred)


def recall(pred, gt):
    """Character-level recall."""
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    pred_counter = Counter(pred)
    gt_counter = Counter(gt)
    correct = sum((pred_counter & gt_counter).values())
    return correct / len(gt)


def f1_score(pred, gt):
    """Character-level F1 score."""
    p = precision(pred, gt)
    r = recall(pred, gt)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# -----------------------------------------------------------------------------
# Evaluation

def evaluate_ocr(model, tokenizer, dataset, device, max_samples=-1, max_tokens=2048, verbose=True):
    """
    Evaluate OCR model on a dataset.
    Returns dict with metrics matching benchmark standards:
    - Fox: precision
    - OmniDocBench: normalized_edit_distance
    """
    import torch
    from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens
    from nanochat.engine import Engine

    image_token_id = tokenizer.encode_special("<|image|>")
    n_img_tokens = count_vision_tokens(base_size=1024)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    engine = Engine(model, tokenizer)
    n_samples = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))

    results = []
    total_ned, total_precision, total_f1 = 0.0, 0.0, 0.0
    n_valid = 0

    for i in range(n_samples):
        sample = dataset[i]
        if not sample.get("images"):
            continue

        image = sample["images"][0]
        gt_text = sample["messages"][1]["content"]

        # Generate
        pixel_values = process_image(image.convert("RGB"), base_size=1024).unsqueeze(0).to(device)
        prompt = "<|image|>\nOCR this document."
        prompt_content = tokenizer.enc.encode(prompt, allowed_special={"<|image|>"})
        prompt_ids = [bos, user_start] + list(prompt_content) + [user_end, assistant_start]
        expanded = expand_image_tokens(prompt_ids, image_token_id, n_img_tokens)

        gen_tokens = []
        with torch.autocast(device, dtype=torch.bfloat16):
            for token_column, _ in engine.generate(expanded, pixel_values=pixel_values,
                                                    max_tokens=max_tokens, temperature=0.0):
                if token_column[0] == assistant_end:
                    break
                gen_tokens.append(token_column[0])
        pred_text = tokenizer.decode(gen_tokens).strip()

        # Compute metrics
        ned = normalized_edit_distance(pred_text, gt_text)
        prec = precision(pred_text, gt_text)
        f1 = f1_score(pred_text, gt_text)

        total_ned += ned
        total_precision += prec
        total_f1 += f1
        n_valid += 1

        if verbose:
            status = "+" if ned < 0.3 else ("~" if ned < 0.6 else "-")
            print(f"{i:<4} ned={ned:.3f} prec={prec:.3f} {status}")

        results.append({
            "index": i, "ned": round(ned, 4), "precision": round(prec, 4),
            "f1": round(f1, 4), "metadata": sample.get("metadata", {}),
        })

    return {
        "num_samples": n_valid,
        "avg_ned": round(total_ned / n_valid, 4) if n_valid else 0,
        "avg_precision": round(total_precision / n_valid, 4) if n_valid else 0,
        "avg_f1": round(total_f1 / n_valid, 4) if n_valid else 0,
        "results": results,
    }


# -----------------------------------------------------------------------------
# Simple eval interface (matches chat_eval.run_chat_eval pattern)

def run_vision_eval(task_name, model, tokenizer, max_problems=None):
    """
    Run vision eval on a task. Returns primary metric (higher is better).
    - Fox: precision
    - OmniDocBench: 1 - NED
    """
    from tasks.fox import Fox
    from tasks.omnidocbench import OmniDocBench

    task_module = {
        'Fox': Fox,
        'OmniDocBench': OmniDocBench,
    }[task_name]
    dataset = task_module()

    device = next(model.parameters()).device
    out = evaluate_ocr(model, tokenizer, dataset, device, max_samples=max_problems or -1, verbose=False)

    if task_name == "Fox":
        return out["avg_precision"]
    else:
        return 1 - out["avg_ned"]
