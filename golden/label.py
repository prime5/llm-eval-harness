#!/usr/bin/env python3
"""P4: blind-labeling tool for the golden conversation set.

Human-only. Nobody else writes these labels.

Shows one conversation at a time with no identifying metadata (no id, tags,
split, construction_note, origin, or judge output), asks for a one-line reason
then an integer score (0-2) per rubric dimension, and appends one row per item
to golden/labels.jsonl. Resumes where you stopped.

Labeling order is shuffled with a stored seed; the 4 repeat items always land
in the second half, so repeat judgments stay blind.

Usage:
    python3 golden/label.py            # resumable

While labeling, type 'anchors' at any prompt to re-read the rubric anchors,
or 'quit' to stop cleanly (progress is saved after every item).
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.rubric import RubricError, load_rubric  # noqa: E402

RUBRIC_PATH = ROOT / "rubrics" / "conversation_v1.yaml"
GOLDEN_PATH = ROOT / "golden" / "conversations_v1.jsonl"
LABELS_PATH = ROOT / "golden" / "labels.jsonl"
ORDER_PATH = ROOT / "golden" / "labeling_order.json"
LABELER = "pramathesh"
ORDER_SEED = 7  # same seed as the dev/test split


def load_items(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_repeat(item_id):
    return item_id.endswith("r")


def build_order(items, seed):
    """Deterministic labeling order: base items shuffled, then repeats.
    The repeats appear in the second half so they can't be spotted as repeats.
    """
    base = [it["id"] for it in items if not is_repeat(it["id"])]
    repeats = [it["id"] for it in items if is_repeat(it["id"])]
    rng = random.Random(seed)
    rng.shuffle(base)
    rng2 = random.Random(seed + 1)
    rng2.shuffle(repeats)
    return base + repeats


def get_order(order_path, items, seed):
    """Return the stored order, creating it the first time."""
    if order_path.exists():
        manifest = json.loads(order_path.read_text())
        if set(manifest["order"]) != {it["id"] for it in items}:
            raise ValueError(
                f"{order_path} does not cover the current golden set; "
                "delete it only if you intend to restart labeling from scratch."
            )
        return manifest["order"]
    order = build_order(items, seed)
    order_path.write_text(json.dumps({"seed": seed, "order": order}, indent=2))
    return order


def already_labeled(labels_path):
    labeled = set()
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            if line.strip():
                labeled.add(json.loads(line)["id"])
    return labeled


def render_conversation(item, number, total):
    """Only the public view: system prompt + turns. Never id/tags/split/notes."""
    lines = [f"\n{'=' * 70}", f"item {number}/{total}", f"{'=' * 70}"]
    system = item.get("system_prompt")
    if system:
        lines.append(f"\n[system] {system}")
    for i, turn in enumerate(item["turns"], start=1):
        lines.append(f"\n--- turn {i} ({turn['role']}) ---")
        lines.append(turn["content"])
    return "\n".join(lines)


def render_anchors(rubric):
    return "\nrubric anchors:\n" + rubric.render()


def parse_score(text, scale):
    """Return the int score, or None if invalid."""
    text = text.strip()
    try:
        value = int(text)
    except ValueError:
        return None
    return value if value in scale else None


class Quit(Exception):
    pass


def ask(prompt, input_fn):
    try:
        text = input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        raise Quit
    if text.strip().lower() in ("quit", "q", "exit"):
        raise Quit
    return text


def collect_item(item, rubric, input_fn, number, total):
    """Collect reasons then scores for one item. Returns (scores, reasons)."""
    print(render_conversation(item, number, total))
    scores, reasons = {}, {}
    scale = sorted(rubric.scale)
    for dim_id in rubric.dimension_ids:
        while True:
            reason = ask(f"\n[{dim_id}] reason (cites turn numbers; 'anchors' to re-read): ", input_fn)
            if reason.strip().lower() == "anchors":
                print(render_anchors(rubric))
                continue
            if not reason.strip():
                print("  reason is required — one line, even if it says the case is ambiguous.")
                continue
            break
        while True:
            raw = ask(f"[{dim_id}] score {scale} (reason first, then decide): ", input_fn)
            if raw.strip().lower() == "anchors":
                print(render_anchors(rubric))
                continue
            score = parse_score(raw, scale)
            if score is None:
                print(f"  enter one of {scale}.")
                continue
            break
        reasons[dim_id] = reason.strip()
        scores[dim_id] = score
    # Final guard: the rubric itself validates dimension coverage and score range.
    return rubric.validate_scores(scores), reasons


def append_label(labels_path, item_id, scores, reasons):
    row = {
        "id": item_id,
        "round": 1,
        "labeler": LABELER,
        "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scores": scores,
        "reasons": reasons,
    }
    with labels_path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
    return row


def run(rubric_path=RUBRIC_PATH, golden_path=GOLDEN_PATH, labels_path=LABELS_PATH,
        order_path=ORDER_PATH, input_fn=input):
    try:
        rubric = load_rubric(rubric_path)
    except RubricError as exc:
        print(f"cannot label without a valid rubric: {exc}", file=sys.stderr)
        return 1
    items = load_items(golden_path)
    by_id = {it["id"]: it for it in items}
    order = get_order(order_path, items, ORDER_SEED)
    done = already_labeled(labels_path)
    pending = [i for i in order if i not in done]

    if not pending:
        print(f"all {len(order)} items already labeled — nothing to do.")
        return 0

    print(f"resuming: {len(done)}/{len(order)} already labeled, {len(pending)} to go.")
    print("Type 'anchors' at any prompt to re-read the rubric; 'quit' to stop (progress saves per item).")
    labeled_now = 0
    try:
        for n, item_id in enumerate(pending, start=len(done) + 1):
            scores, reasons = collect_item(by_id[item_id], rubric, input_fn, n, len(order))
            append_label(labels_path, item_id, scores, reasons)
            labeled_now += 1
    except Quit:
        pass

    total = len(done) + labeled_now
    if total == len(order):
        print(f"\ndone: {total}/{len(order)} labeled. Commit: golden: human labels round 1")
    else:
        print(f"\nstopped at {total}/{len(order)} (partial {total}/28). Commit: golden: human labels round 1 (partial {total}/28)")
    return 0


def main():
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
