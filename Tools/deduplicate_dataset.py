"""
MinHash deduplication for historical_regex_keyword_gpt.csv / modern_regex_keyword_gpt.csv.

Usage:
    python3.11 deduplicate_dataset.py --historical
    python3.11 deduplicate_dataset.py --modern
    python3.11 deduplicate_dataset.py --src-file /path/to/input.csv --dst-file /path/to/output.csv

Methodology follows HuggingFace cosmopedia approach (4-stage MinHash LSH deduplication),
adapted from SlurmPipelineExecutor → LocalPipelineExecutor for single-machine use.
"""

import os
import json
import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import CSVReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

DATASETS_DIR = "/home/ec2-user/wxchat/FastText-trainer-finewebedu-climate/datasets"
WORK_DIR = f"{DATASETS_DIR}/minhash_work"  # Temp working directory for minhash stages

# Single machine — no need for many tasks
TOTAL_TASKS = 1

minhash_config = MinhashConfig()


def build_pipeline(src_file: str, dst_file: str):
    """Build and run the 4-stage MinHash dedup pipeline, then convert JSONL → CSV."""
    src_folder = os.path.dirname(src_file)
    src_filename = os.path.basename(src_file)
    dataset_name = os.path.splitext(src_filename)[0]

    base_path = f"{WORK_DIR}/{dataset_name}"
    sig_path = f"{base_path}/signatures"
    bucket_path = f"{base_path}/buckets"
    cluster_path = f"{base_path}/remove_ids"
    dedup_output_path = f"{base_path}/deduplicated"
    log_path = f"{base_path}/logs"

    os.makedirs(WORK_DIR, exist_ok=True)

    input_reader = CSVReader(
        data_folder=src_folder,
        glob_pattern=src_filename,
        text_key="Text",   # our column name
        id_key="Date",     # our column name
    )

    # Stage 1: Compute MinHash signatures
    stage1 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            MinhashDedupSignature(output_folder=sig_path, config=minhash_config),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{log_path}/signatures",
    )

    # Stage 2: Find matches between signatures in each bucket
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=sig_path,
                output_folder=bucket_path,
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        logging_dir=f"{log_path}/buckets",
        depends=stage1,
    )

    # Stage 3: Cluster duplicates
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=bucket_path,
                output_folder=cluster_path,
                config=minhash_config,
            ),
        ],
        tasks=1,
        logging_dir=f"{log_path}/clusters",
        depends=stage2,
    )

    # Stage 4: Filter — keep only one sample per duplicate cluster
    stage4 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            MinhashDedupFilter(
                input_folder=cluster_path,
                exclusion_writer=JsonlWriter(f"{base_path}/removed"),
            ),
            JsonlWriter(output_folder=dedup_output_path),
        ],
        tasks=TOTAL_TASKS,
        logging_dir=f"{log_path}/filter",
        depends=stage3,
    )

    print(f"\nRunning 4-stage MinHash dedup on: {src_file}")
    stage4.run()

    # Convert JSONL output → CSV
    print(f"\nConverting JSONL → CSV: {dst_file}")
    jsonl_to_csv(dedup_output_path, dst_file)
    print(f"Done. Deduplicated CSV saved to: {dst_file}")


def jsonl_to_csv(jsonl_folder: str, dst_csv: str):
    """Collect all JSONL files from dedup output and write as Date,Text CSV."""
    rows = []
    for fname in sorted(os.listdir(jsonl_folder)):
        if not fname.endswith(".jsonl") and not fname.endswith(".jsonl.gz"):
            continue
        fpath = os.path.join(jsonl_folder, fname)
        open_fn = open
        if fname.endswith(".gz"):
            import gzip
            open_fn = lambda p, **kw: gzip.open(p, "rt", encoding="utf-8")
        with open_fn(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                date = doc.get("id", doc.get("Date", ""))
                text = doc.get("text", doc.get("Text", ""))
                rows.append([date, text])

    before_count_file = dst_csv.replace("_deduped.csv", "_keyword_gpt.csv")
    print(f"  Deduplicated rows: {len(rows)}")

    with open(dst_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Date", "Text"])
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MinHash deduplication (cosmopedia methodology, local execution)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--historical", action="store_true",
                        help="Dedup historical_regex_keyword_gpt.csv")
    parser.add_argument("--modern", action="store_true",
                        help="Dedup modern_regex_keyword_gpt.csv")
    parser.add_argument("--src-file", help="Source CSV file path", default=None)
    parser.add_argument("--dst-file", help="Destination CSV file path", default=None)
    args = parser.parse_args()

    if args.historical:
        src = f"{DATASETS_DIR}/historical_regex_keyword_gpt.csv"
        dst = f"{DATASETS_DIR}/historical_regex_keyword_gpt_deduped.csv"
    elif args.modern:
        src = f"{DATASETS_DIR}/modern_regex_keyword_gpt.csv"
        dst = f"{DATASETS_DIR}/modern_regex_keyword_gpt_deduped.csv"
    elif args.src_file and args.dst_file:
        src = args.src_file
        dst = args.dst_file
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Source:      {src}")
    print(f"Destination: {dst}")
    build_pipeline(src, dst)
