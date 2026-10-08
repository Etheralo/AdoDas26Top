#!/usr/bin/env python3
"""Extract token-level embeddings of ASR transcripts with a frozen Chinese encoder.

The transcript file format is not fixed yet, so columns are configurable. Input
is a CSV / TSV / JSONL / parquet table with one row per (participant, session):

    python tools/extract_text_embeddings.py \
        --transcripts /data/asr/transcripts.csv --text_root /data/text_emb \
        --col_split split --col_text text

Output: <text_root>/<split>/<school>/<class>/<pid>/text/<tag>/<session>/sequence.npz
with ``features`` (L, D) float16 and ``valid_mask`` (L,). Texts longer than the
encoder limit are split into chunks and the token states concatenated.
Transcripts and embeddings stay on local disk; do not commit them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.text_io import text_path  # noqa: E402


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=suffix == ".jsonl", dtype=False)
    return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str, keep_default_na=False)


@torch.no_grad()
def encode(text: str, tokenizer, model, device, max_length: int, n_layers: int) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return np.zeros((0, model.config.hidden_size), dtype=np.float16)
    step = max_length - 2
    outs = []
    for start in range(0, len(ids), step):
        chunk = tokenizer.build_inputs_with_special_tokens(ids[start:start + step])
        inp = torch.tensor([chunk], device=device)
        hs = model(input_ids=inp, attention_mask=torch.ones_like(inp), output_hidden_states=True).hidden_states
        h = torch.stack(hs[-n_layers:]).mean(0)[0, 1:-1]  # drop [CLS]/[SEP] of each chunk
        outs.append(h.float().cpu().numpy())
    return np.concatenate(outs).astype(np.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcripts", help="CSV/TSV/JSONL/parquet table")
    source.add_argument("--transcript_root", help="Extracted official transcript tree")
    ap.add_argument("--manifest_dir", help="Required with --transcript_root")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test_hidden"])
    ap.add_argument("--limit", type=int, default=0, help="Diagnostic record limit; 0 means all")
    ap.add_argument("--text_root", required=True)
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext-large")
    ap.add_argument("--revision", default="main", help="Pin a Hub commit for reproducible extraction")
    ap.add_argument("--tag", default="chinese-roberta-wwm-ext-large", help="Must match features.text_tag")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4, help="Average of the last N hidden layers")
    ap.add_argument("--split", default=None, help="Use this split for every row instead of --col_split")
    ap.add_argument("--col_split", default="split")
    ap.add_argument("--col_school", default="anon_school")
    ap.add_argument("--col_class", default="anon_class")
    ap.add_argument("--col_pid", default="anon_pid")
    ap.add_argument("--col_session", default="session")
    ap.add_argument("--col_text", default="text")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    if args.transcript_root:
        if not args.manifest_dir:
            ap.error("--transcript_root requires --manifest_dir")
        # Yield one private text at a time; never write transcript tables or log texts.
        def records():
            from data.dataset import read_manifest
            root = Path(args.transcript_root)
            for split in args.splits:
                manifest = Path(args.manifest_dir) / f"{split}.csv"
                if not manifest.exists():
                    continue
                for row in read_manifest(manifest).to_dict("records"):
                    if row["session"] not in ("B01", "B02", "B03"):
                        continue
                    directory = root / split / row["anon_school"] / row["anon_class"] / row["anon_pid"] / row["session"]
                    path = directory / "normalized_transcript.txt"
                    if not path.exists():
                        path = directory / "clean_transcript.txt"
                    if not path.exists():
                        raise FileNotFoundError("Transcript missing from official tree")
                    yield {args.col_split: split, args.col_school: row["anon_school"],
                           args.col_class: row["anon_class"], args.col_pid: row["anon_pid"],
                           args.col_session: row["session"], args.col_text: path.read_text(encoding="utf-8")}
        rows = records()
    else:
        df = read_table(Path(args.transcripts))
        rows = (row for _, row in df.iterrows())
    if not args.transcript_root:
        needed = [args.col_school, args.col_class, args.col_pid, args.col_session, args.col_text]
        if args.split is None:
            needed.append(args.col_split)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Transcript table lacks columns {missing}; available: {list(df.columns)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # User policy: use local files/cache only; never download model weights.
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, revision=args.revision, local_files_only=True).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    written = skipped = empty = 0
    for record_index, row in enumerate(rows):
        if args.limit and record_index >= args.limit:
            break
        split = args.split or str(row[args.col_split])
        out = text_path(args.text_root, split, str(row[args.col_school]), str(row[args.col_class]),
                        str(row[args.col_pid]), args.tag, str(row[args.col_session]))
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        text = str(row[args.col_text] or "").strip()
        feats = encode(text, tokenizer, model, device, args.max_length, args.layers)
        if len(feats) == 0:
            empty += 1
            # An empty transcript is valid missing evidence. Write an explicit
            # empty record so coverage checks distinguish it from unfinished extraction.
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_name("sequence.tmp.npz")
        np.savez_compressed(temporary, features=feats, valid_mask=np.ones(len(feats), dtype=bool),
                            model_name=np.asarray(args.model),
                            model_revision=np.asarray(getattr(model.config, "_commit_hash", None) or args.revision))
        temporary.replace(out)
        written += 1
        if written % 500 == 0:
            print(f"{written} written")
    summary = {"model": args.model, "revision": getattr(model.config, "_commit_hash", None) or args.revision,
               "layers": args.layers, "chunk_length": args.max_length, "written": written,
               "skipped_existing": skipped, "empty_text": empty, "splits": args.splits}
    Path(args.text_root).mkdir(parents=True, exist_ok=True)
    (Path(args.text_root) / "extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written {written}, skipped (exists) {skipped}, empty text {empty}")


if __name__ == "__main__":
    main()
