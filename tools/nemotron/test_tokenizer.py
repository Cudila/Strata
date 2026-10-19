#!/usr/bin/env python3
# tools/nemotron/test_tokenizer.py - the server's tokenizer (tools/strata_tokenizer.py with the pack's `pre`) against
# the model's own HF tokenizer.json: the same ids for plain text (every line of wikitext-2 test) and for rendered chat
# prompts with their control tokens (<|im_start|>, <|im_end|>, <think>, <tool_call>), and decode(encode(x)) == x.
#
# Usage: python test_tokenizer.py <pack tokenizer dir> <HF tokenizer.json> [text file]
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import strata_tokenizer as ST  # noqa: E402
from tokenizers import Tokenizer as HF  # noqa: E402


def load(d):
    vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = open(os.path.join(d, "merges.txt"), encoding="utf-8").read().split("\n")
    types = json.load(open(os.path.join(d, "token_type.json")))
    cfg = json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, pre=cfg.get("pre", "qwen35"))


def main():
    ours = load(sys.argv[1])
    hf = HF.from_file(sys.argv[2])
    text = open(sys.argv[3], encoding="utf-8").read()
    lines = [l for l in text.split("\n") if l.strip()]
    extra = ["Привет, мир! Как дела? Ёлки-палки, 123456 и 3.14159.", "def f(x):\n    return x**2  # квадрат\n",
             "日本語のテキストと中文混合 with English", "émigré naïve façade — «кавычки» “quotes”", "\t\ttabs\r\nCRLF  \n\n"]
    bad = 0
    for s in lines + extra:
        a = ours.encode(s, parse_special=False)
        b = hf.encode(s, add_special_tokens=False).ids
        if a != b:
            bad += 1
            if bad <= 3:
                print("MISMATCH plain:", repr(s[:80]), a[:12], b[:12])
        if ours.decode(a) != s:
            bad += 1
            if bad <= 3:
                print("ROUND-TRIP:", repr(s[:80]))
    chats = [
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nПривет!<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>assistant\n<think></think><tool_call>\n<function=get_weather>\n<parameter=city>\nKazan\n</parameter>\n"
        "</function>\n</tool_call>\n<|im_end|>\n",
    ]
    for s in chats:
        a = ours.encode(s, parse_special=True)
        b = hf.encode(s, add_special_tokens=False).ids
        if a != b:
            bad += 1
            print("MISMATCH chat:", repr(s[:80]), a[:16], b[:16])
    print(f"{len(lines) + len(extra)} plain strings, {len(chats)} chat prompts: {bad} mismatches")
    print("PASS" if bad == 0 else "FAIL")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
