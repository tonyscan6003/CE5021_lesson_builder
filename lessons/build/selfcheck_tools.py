#!/usr/bin/env python3
"""Write, extend and validate the self-check YAML files in content/_selfcheck/.

Exists because yaml.dump() on a plain scalar appends YAML's document-end marker
("..."), which once silently corrupted eight of these files. Everything here
writes through `_scalar()`, which cannot produce that marker, and `validate()`
re-parses every file it writes.

    python lessons/build/selfcheck_tools.py --validate
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

SELFCHECK = Path(__file__).resolve().parents[1] / "content" / "_selfcheck"
KEYS = {"q", "options", "answer", "explain"}


def _scalar(v) -> str:
    """One YAML scalar, correctly quoted, with no document-end marker."""
    return yaml.safe_dump({"k": v}, allow_unicode=True, width=10 ** 6,
                          default_flow_style=False)[3:].strip()


def _header(text: str) -> str:
    return "".join(l for l in text.splitlines(keepends=True) if l.startswith("#"))


def dump(items) -> str:
    out = []
    for it in items:
        out.append(f"- q: {_scalar(it['q'])}")
        out.append("  options:")
        out += [f"    - {_scalar(o)}" for o in it["options"]]
        out.append(f"  answer: {it['answer']}")
        out.append(f"  explain: {_scalar(it['explain'])}")
    return "\n".join(out) + "\n"


def shuffle_options(items, seed):
    """Shuffle each question's options deterministically; True/False keep order."""
    rng = random.Random(seed)
    for it in items:
        opts = it["options"]
        if [str(o).strip().lower() for o in opts] == ["true", "false"]:
            continue
        correct = opts[it["answer"]]
        order = list(range(len(opts)))
        rng.shuffle(order)
        it["options"] = [opts[i] for i in order]
        it["answer"] = it["options"].index(correct)
    return items


def write(lesson_id: str, header: str, items, *, shuffle_from: int = 0, seed=None):
    """Write a self-check file. Items from index `shuffle_from` onward get their
    options shuffled (earlier ones are assumed already shuffled)."""
    if shuffle_from < len(items):
        shuffle_options(items[shuffle_from:], seed or f"{lesson_id}:{shuffle_from}")
    path = SELFCHECK / f"{lesson_id}.yaml"
    path.write_text(header.rstrip("\n") + "\n" + dump(items), encoding="utf-8")
    validate_file(path)
    return path


def extend(lesson_id: str, new_items, new_header: str | None = None):
    """Append questions to an existing file, shuffling only the new ones."""
    path = SELFCHECK / f"{lesson_id}.yaml"
    text = path.read_text(encoding="utf-8")
    items = yaml.safe_load(text) or []
    n_old = len(items)
    return write(lesson_id, new_header or _header(text), items + list(new_items),
                 shuffle_from=n_old)


def validate_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    assert "\n...\n" not in text and not text.startswith("..."), f"{path.name}: stray '...'"
    items = yaml.safe_load(text)
    assert isinstance(items, list) and items, f"{path.name}: not a non-empty list"
    for i, it in enumerate(items):
        assert set(it) == KEYS, (path.name, i, set(it))
        assert isinstance(it["options"], list) and len(it["options"]) >= 2, (path.name, i)
        assert isinstance(it["answer"], int) and 0 <= it["answer"] < len(it["options"]), \
            (path.name, i)
        assert str(it["q"]).strip() and str(it["explain"]).strip(), (path.name, i)
    return len(items)


def validate() -> dict:
    return {p.stem: validate_file(p) for p in sorted(SELFCHECK.glob("*.yaml"))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    if ap.parse_args().validate or True:
        counts = validate()
        for k, v in counts.items():
            print(f"  {k}: {v} questions")
        print(f"{sum(counts.values())} questions in {len(counts)} files — all valid")
