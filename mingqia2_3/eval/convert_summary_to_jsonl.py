#!/usr/bin/env python3
"""
Convert a simulation_summary.json (from make_requests.py) into a student_outputs.jsonl-style file.

Each line of the output JSONL is:
{"index": <prompt_idx>, "prompt": "<original prompt or ''>", "text": "<model text or error>"}

If you pass --arrivals pointing to batch_arrivals.json, prompts will be populated from there.
Otherwise prompts default to empty strings.
"""
import argparse
import json
from pathlib import Path


def load_prompt_map(arrivals_path: Path) -> dict:
    """Return a mapping from prompt_idx -> prompt from batch_arrivals.json."""
    prompt_map = {}
    batches = json.loads(arrivals_path.read_text())
    for batch in batches:
        for idx, prompt in zip(batch.get("prompt_idxs", []), batch.get("prompts", [])):
            prompt_map[idx] = prompt
    return prompt_map


def convert(summary_path: Path, output_path: Path, prompt_map: dict) -> None:
    summary = json.loads(summary_path.read_text())
    results = summary.get("results", [])
    outputs = {}  # idx -> {"prompt": ..., "text": ...}

    for batch in results:
        prompt_idxs = batch.get("prompt_idxs", [])
        status = batch.get("status_code")
        error = batch.get("error")
        choices = (batch.get("response") or {}).get("choices") or []

        if status != 200 or not choices:
            err_text = error or f"HTTP {status}"
            for i, idx in enumerate(prompt_idxs):
                # Try to recover prompt from choices if present, otherwise arrivals map
                prompt_val = ""
                if choices and i < len(choices):
                    prompt_val = choices[i].get("prompt", "") or ""
                if not prompt_val:
                    prompt_val = prompt_map.get(idx, "")
                if not prompt_val:
                    prompt_val = "<prompt unavailable>"
                outputs.setdefault(idx, {"prompt": prompt_val, "text": err_text})
            continue

        texts = [c.get("text", "") for c in choices]
        prompts_from_choices = [c.get("prompt", "") for c in choices]
        if len(texts) < len(prompt_idxs):
            texts.extend([""] * (len(prompt_idxs) - len(texts)))
            prompts_from_choices.extend([""] * (len(prompt_idxs) - len(prompts_from_choices)))
        texts = texts[: len(prompt_idxs)]
        prompts_from_choices = prompts_from_choices[: len(prompt_idxs)]
        for idx, text, ch_prompt in zip(prompt_idxs, texts, prompts_from_choices):
            prompt_val = ch_prompt or prompt_map.get(idx, "")
            outputs.setdefault(idx, {"prompt": prompt_val, "text": text})

    with output_path.open("w") as f:
        for idx in sorted(outputs.keys()):
            record = {"index": idx, "prompt": outputs[idx]["prompt"], "text": outputs[idx]["text"]}
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(outputs)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert simulation_summary.json to student_outputs.jsonl")
    parser.add_argument("--summary", required=True, help="Path to simulation_summary.json")
    parser.add_argument(
        "--arrivals",
        help="Optional path to batch_arrivals.json to include original prompts",
    )
    parser.add_argument("--output", default="student_outputs.jsonl", help="Path to write JSONL")
    args = parser.parse_args()
    prompt_map = {}
    if args.arrivals:
        prompt_map = load_prompt_map(Path(args.arrivals))
    else:
        default_arrivals = Path(args.summary).parent / "batch_arrivals.json"
        if default_arrivals.exists():
            prompt_map = load_prompt_map(default_arrivals)
    convert(Path(args.summary), Path(args.output), prompt_map)


if __name__ == "__main__":
    main()
