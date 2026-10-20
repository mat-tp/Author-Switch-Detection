#!/usr/bin/env python3
"""
Builds a PSEUDO dataset in the MAWSA folder layout, from public-domain books.

    *** THIS IS NOT PAN DATA. ***
    It exists only so the whole pipeline (feature extraction, the three models, the ablation
    study, the retrieval model, the web app and the TIRA runner) can be run and tested on a machine that
    does not have the Zenodo download. Numbers measured on it say nothing about the PAN 2026
    task and must never be reported as results of the shared task.

How it is made: every "author" is one book (NLTK's Gutenberg sample). A problem is a list of
consecutive sentences taken from segments of 2 to 5 different books; the label of a boundary is 1
when the book changes. The three difficulty levels only change the length of the segments
(easy: long segments, medium: shorter, hard: 1 to 3 sentences). The train and validation
parts use different regions of every book, so no sentence is in both.

Usage:
    python3 scripts/make_pseudo_mawsa.py GUTENBERG_DIR OUT_DIR [--train 400] [--val 100] [--seed 7]

    GUTENBERG_DIR  folder with the *.txt books (e.g. nltk_data/corpora/gutenberg)
    OUT_DIR        e.g. dataset/pseudo-mawsa/  (then:  java -Ddataset=dataset/pseudo-mawsa/ ... Main)
"""
import argparse
import json
import os
import random
import re

SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=["\'(\[]?[A-Z])')

# (min segment length, max segment length) in sentences
DIFFICULTY = {"easy": (8, 20), "medium": (3, 8), "hard": (1, 3)}


def load_sentences(path):
    text = open(path, encoding="latin-1").read()
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [Moby Dick by Herman Melville 1851]
    text = re.sub(r"\s+", " ", text)
    sentences = []
    for s in SENTENCE_END.split(text):
        s = s.strip()
        n_words = len(s.split())
        if 4 <= n_words <= 60 and not s.isupper():
            sentences.append(s)
    return sentences


def make_problem(rng, books, difficulty):
    lo, hi = DIFFICULTY[difficulty]
    n_authors = rng.randint(2, 5)
    chosen = rng.sample(sorted(books), n_authors)
    target = rng.randint(24, 60)

    sentences, owners = [], []
    previous = None
    while len(sentences) < target:
        author = rng.choice([a for a in chosen if a != previous])
        length = rng.randint(lo, hi)
        pool = books[author]
        start = rng.randrange(0, max(1, len(pool) - length))
        for s in pool[start:start + length]:
            sentences.append(s)
            owners.append(author)
        previous = author

    changes = [1 if owners[i] != owners[i + 1] else 0 for i in range(len(owners) - 1)]
    return sentences, changes, len(set(owners))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gutenberg_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--train", type=int, default=400, help="problems per difficulty, train")
    parser.add_argument("--val", type=int, default=100, help="problems per difficulty, validation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unseen", type=int, default=0,
                        help="hold out this many books completely: they appear ONLY in validation (authors "
                             "the models never saw, like the authors of new PAN documents)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    train_books, val_books = {}, {}
    all_books = {}
    for name in sorted(os.listdir(args.gutenberg_dir)):
        if not name.endswith(".txt"):
            continue
        sentences = load_sentences(os.path.join(args.gutenberg_dir, name))
        if len(sentences) < 400:
            continue
        all_books[name] = sentences

    held_out = set(rng.sample(sorted(all_books), args.unseen)) if args.unseen else set()
    for name, sentences in all_books.items():
        if name in held_out:
            val_books[name] = sentences  # never in train
            continue
        cut = int(len(sentences) * 0.7)
        train_books[name] = sentences[:cut]
        if not args.unseen:
            val_books[name] = sentences[cut:]
    if args.unseen:
        print("held out (validation only):", sorted(held_out))

    print("books:", len(train_books))

    for difficulty in DIFFICULTY:
        for split, books, count in (("train", train_books, args.train), ("validation", val_books, args.val)):
            folder = os.path.join(args.out_dir, difficulty, split)
            os.makedirs(folder, exist_ok=True)
            n_pairs = n_changes = 0
            for k in range(1, count + 1):
                sentences, changes, authors = make_problem(rng, books, difficulty)
                with open(os.path.join(folder, "problem-%d.txt" % k), "w", encoding="utf-8") as f:
                    f.write("\n".join(sentences) + "\n")
                with open(os.path.join(folder, "truth-problem-%d.json" % k), "w", encoding="utf-8") as f:
                    json.dump({"authors": authors, "changes": changes}, f)
                n_pairs += len(changes)
                n_changes += sum(changes)
            print("%-6s %-10s %4d problems, %6d pairs, %.1f%% changes" % (
                difficulty, split, count, n_pairs, 100.0 * n_changes / max(1, n_pairs)))

    with open(os.path.join(args.out_dir, "NOT_PAN_DATA.txt"), "w") as f:
        f.write("Pseudo dataset made by scripts/make_pseudo_mawsa.py from public domain books.\n"
                "It is NOT the PAN 2026 data. Use it only to test the software.\n")


if __name__ == "__main__":
    main()
