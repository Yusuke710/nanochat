"""Vision evaluation - generate completions and compute task metrics."""

import torch

def evaluate_vision_task(model, tokenizer, dataset, device, max_samples=-1, batch_size=8):
    """Run vision task evaluation with batched generation. Returns dict with avg_score and results."""
    from nanochat.image_process import process_image, count_vision_tokens, expand_image_tokens
    from nanochat.engine import Engine

    enc = tokenizer.enc
    tok = lambda s: tokenizer.encode_special(s)
    image_tok, n_img_tok = tok("<|image|>"), count_vision_tokens(base_size=1024)

    engine = Engine(model, tokenizer)
    n = len(dataset) if max_samples < 0 else min(max_samples, len(dataset))

    # Collect all samples first
    samples, prompts, pixel_values = [], [], []
    for i in range(n):
        sample = dataset[i]
        if not sample.get("images"):
            continue

        # build prompt: [bos, user_start, content, user_end, assistant_start]
        prompt = sample["messages"][0]["content"].replace("<image>", "<|image|>")
        ids = [tokenizer.get_bos_token_id(), tok("<|user_start|>")]
        ids += list(enc.encode(prompt, allowed_special={"<|image|>"}))
        ids += [tok("<|user_end|>"), tok("<|assistant_start|>")]
        ids = expand_image_tokens(ids, image_tok, n_img_tok)

        pix = process_image(sample["images"][0].convert("RGB"), base_size=1024).unsqueeze(0).to(device)

        samples.append((i, sample))
        prompts.append(ids)
        pixel_values.append(pix)

    # Process in batches
    results, total = [], 0.0
    for b in range(0, len(samples), batch_size):
        batch_samples = samples[b:b + batch_size]
        batch_prompts = prompts[b:b + batch_size]
        batch_pix = pixel_values[b:b + batch_size]

        with torch.autocast(device.type, dtype=torch.bfloat16):
            batch_tokens = engine.generate_prompts(batch_prompts, batch_pix, max_tokens=2048, temperature=0.0)

        for (i, sample), tokens in zip(batch_samples, batch_tokens):
            pred = tokenizer.decode(tokens).strip()
            score = dataset.evaluate(sample, pred)
            total += score
            print(f"{i:4d} {'+' if score > 0.7 else '-'} {score:.3f}")
            results.append({"idx": i, "score": score, "pred": pred, "gt": sample["messages"][1]["content"]})

    avg = total / len(results) if results else 0
    return {"avg_score": round(avg, 4), "num_samples": len(results), "results": results}
