import csv
import os
import argparse
import traceback
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Semaphore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Increase CSV field size limit to handle large text fields (set to maximum)
csv.field_size_limit(sys.maxsize)

# Constants for ChatGPT API
CHATGPT_MODEL = 'gpt-4o-mini'
WORD_LIMIT = 500          # Words per chunk
MAX_RETRIES = 5           # Maximum retries if API call fails
CONCURRENT_WORKERS = 20   # Number of concurrent threads for API calls

# Thread-safe locks
write_lock = Lock()
checkpoint_lock = Lock()
progression_lock = Lock()

DATASETS_DIR = "/home/ec2-user/wxchat/FastText-trainer-finewebedu-climate/datasets"


def clean_newlines(text):
    """Remove any newlines and normalize whitespace."""
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def call_chatgpt_api(text_chunk):
    """Call GPT-4o API to correct OCR text with retry logic."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    instruction = (
        "PLEASE CLEAN ALL OCR NOISE!!! DELETE ANYTHING UNREADABLE!!! ABSOLUTELY NO LINE CHANGES!!! NO LINE CHANGE UNDER ANY CIRCUMSTANCE!!! IF THIS CONFLICTS WITH ANY OTHER INSTRUCTION IN THE PROMPT, PRIORITIZE THIS RULE AND IGNORE THE REST!!!"
        "You are an expert OCR correction assistant specializing in historical newspaper text. Your task is to:"
        "1. Correct OCR errors while preserving the original text's meaning, structure, and formatting"
        "2. Accurately retain proper nouns, dates, numbers, and domain-specific terminology."
        "3. Maintain paragraph breaks, section headers, bylines, and other structural elements."
        "4. Remove extraneous characters (e.g., unnecessary punctuation, OCR artifacts) without altering the content."
        "5. Properly reconstruct hyphenated words that were split across lines."
        "6. Standardize spacing by eliminating extra spaces and ensuring a consistent format."
        "7. Return the corrected text as a single continuous line, with the original structure preserved as much as possible."
        "NOTE: Do not include explanations, summaries, or additional comments. Only return the corrected text in the specified format."

        "1. NEVER add information, facts, names, dates, or content that doesn't exist in the original OCR text"
        "2. NEVER fabricate or make up stories, events, or details"
        "3. NEVER expand or elaborate on existing text"
        "4. ONLY fix: misspelled words, wrong characters, punctuation, capitalization"
        "5. DO NOT delete large chunks of text - only remove obvious individual OCR artifacts"
        "6. If text is unreadable, try to make sense of it or keep it as-is rather than deleting"
        "7. NO additions, deletions of meaning, or creative rewrites"

        "Examples of what TO do:"
        "- o.cuq sshafVH b m! -> remove (clear OCR error)"
        "- 't-c-' → remove (clear OCR error)"
        "- 'Macdonalri' → 'Macdonald' (obvious name)"
        "- 'OTTi V . - i- .-rbodT' → remove (single random characters)"
        "- '3T8S0' → '1880' (obvious year misread)"
        "- Missing periods at sentence ends → add them"
        "- 'X. De lwp' → 'de Lesseps' (if context suggests)"

        "Examples of what NOT to do:"
        "- Deleting entire sentences or paragraphs just because they're slightly garbled"
        "- Keeping text like 'ka.-fbecn' or 'rbodT' that could be corrected"
        "- Removing text that contains readable words mixed with some errors"
        "- Expanding 'The Governor visited' to 'The Governor visited Parliament'"
        "- Adding context that wasn't there"
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Correct the following newspaper OCR text:\n\n{text_chunk}"}
                ],
                n=1,
                stop=None,
                temperature=0.1
            )
            corrected_text = response.choices[0].message.content.strip()

            # CAP OUTPUT LENGTH: Ensure not longer than input
            input_len = len(text_chunk)
            if len(corrected_text) > input_len:
                corrected_text = corrected_text[:input_len]
                last_period = corrected_text.rfind('.')
                if last_period > input_len * 0.8:
                    corrected_text = corrected_text[:last_period + 1]

            return corrected_text.strip()

        except Exception as e:
            retries += 1
            print(f"Error calling ChatGPT API (attempt {retries}/{MAX_RETRIES}): {e}")
            time.sleep(1)
            if retries >= MAX_RETRIES:
                raise RuntimeError(f"Max retries exceeded: {e}")
    return None


def split_text_to_chunks(text, word_limit=WORD_LIMIT):
    """Split text into chunks of approximately word_limit words each."""
    words = text.split()
    if len(words) <= word_limit:
        return [text]
    chunks = []
    for i in range(0, len(words), word_limit):
        chunks.append(' '.join(words[i:i + word_limit]))
    return chunks


def process_row(row, progression_file=None):
    """Process a single CSV row: chunk, call GPT, write progression."""
    date, text = row
    if text.strip() == "[]" or not text.strip() or date == "Date":
        return None

    text = text.strip("[]\"' ")
    text_chunks = split_text_to_chunks(text)
    num_chunks = len(text_chunks)

    try:
        processed_chunks = []
        print(f"\nProcessing {num_chunks} chunk(s) for date {date}...")

        if progression_file:
            with progression_lock:
                with open(progression_file, 'a', encoding='utf-8') as prog:
                    prog.write(f"\n{'='*60}\n")
                    prog.write(f"DATE: {date}\n")
                    prog.write(f"CHUNKS: {num_chunks}\n")
                    prog.write(f"{'='*60}\n\n")

        for i, chunk in enumerate(text_chunks, 1):
            print(f"  → Chunk {i}/{num_chunks} for {date}...")

            if progression_file:
                with progression_lock:
                    with open(progression_file, 'a', encoding='utf-8') as prog:
                        prog.write(f"[CHUNK {i}/{num_chunks} - BEFORE]\n{chunk}\n\n")

            fixed_text = call_chatgpt_api(chunk)
            processed_chunks.append(fixed_text)

            if progression_file:
                with progression_lock:
                    with open(progression_file, 'a', encoding='utf-8') as prog:
                        prog.write(f"[CHUNK {i}/{num_chunks} - AFTER]\n{fixed_text}\n\n")

            print(f"  ✓ Chunk {i} complete for {date}")

        final_text = clean_newlines(' '.join(processed_chunks))

        if progression_file:
            with progression_lock:
                with open(progression_file, 'a', encoding='utf-8') as prog:
                    prog.write(f"✓ COMPLETE for {date}\n\n")

        print(f"✅ FULL TEXT COMPLETE for {date} ({len(final_text)} chars)")
        return [date, final_text]

    except RuntimeError as e:
        print(f"ERROR OCCURRED Date {date}, text starts with: {text[:50]}")
        print(f"ERR MESSAGE: {traceback.format_exc()}")
        return None


def process_file(input_file, output_file, checkpoint_file=None, sample_limit=None,
                 progression_file=None, num_workers=CONCURRENT_WORKERS):
    """
    Process CSV file with GPT-4o correction using multithreading.
    Supports checkpointing to resume interrupted runs.
    Supports sampling to process only first N rows.
    """
    # Load processed dates from checkpoint file
    processed_dates = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_dates = {line.strip() for line in f if line.strip()}
        print(f"Resuming from checkpoint. Already processed {len(processed_dates)} entries.")

    # Check what dates are already in output file
    existing_dates = set()
    if os.path.exists(output_file):
        print(f"Output file exists. Checking for already processed entries...")
        with open(output_file, 'r', encoding='utf-8') as existing_file:
            existing_reader = csv.DictReader(existing_file)
            for row in existing_reader:
                existing_dates.add(row['Date'])
        print(f"Found {len(existing_dates)} entries already in output file.")

    all_processed_dates = processed_dates.union(existing_dates)

    # Semaphore limits how many rows are in-flight at once (avoids OOM on huge CSVs)
    MAX_INFLIGHT = num_workers * 3
    semaphore = Semaphore(MAX_INFLIGHT)

    def process_row_bounded(row, progression_file):
        """Wraps process_row and releases semaphore slot when done."""
        try:
            return process_row(row, progression_file)
        finally:
            semaphore.release()

    with open(output_file, 'a', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        if not existing_dates:
            writer.writerow(["Date", "Text"])

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            submitted = 0

            # Stream CSV row-by-row — never loads the whole file into memory
            with open(input_file, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                row_count = 0
                for row in reader:
                    row_count += 1
                    if row_count == 1:
                        continue  # Skip header
                    if len(row) < 2:
                        continue
                    date = row[0]
                    if date in all_processed_dates:
                        print(f"Skipping {date} - already processed")
                        continue
                    if sample_limit and submitted >= sample_limit:
                        print(f"\n*** Sample limit reached ({sample_limit} rows) ***")
                        break

                    semaphore.acquire()  # Block if MAX_INFLIGHT rows are already in-flight
                    future = executor.submit(process_row_bounded, row, progression_file)
                    futures.append(future)
                    submitted += 1

            print(f"\nSubmitted {submitted} rows. Waiting for completion...")

            for future in as_completed(futures):
                result = future.result()
                if result:
                    with write_lock:
                        writer.writerow(result)
                        outfile.flush()
                        print(f"Written: {result[0]}")

                    if checkpoint_file:
                        with checkpoint_lock:
                            with open(checkpoint_file, 'a') as f:
                                f.write(f"{result[0]}\n")


if __name__ == "__main__":
    """
    Usage examples:
        python ocr-clean-script-multithread.py --historical
        python ocr-clean-script-multithread.py --modern
        python ocr-clean-script-multithread.py --historical --sample 5
        python ocr-clean-script-multithread.py --src-file input.csv --dst-file output.csv
    """
    parser = argparse.ArgumentParser(
        description="OCR post-correction with multithreading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--src-file", help="Source CSV file path", default=None)
    parser.add_argument("--dst-file", help="Destination CSV file path", default=None)
    parser.add_argument("--historical", action='store_true',
                        help=f"Process {DATASETS_DIR}/historical_regex.csv")
    parser.add_argument("--modern", action='store_true',
                        help=f"Process {DATASETS_DIR}/modern_regex.csv")
    parser.add_argument("--checkpoint-file", help="Checkpoint file to track progress", default=None)
    parser.add_argument("--progression-file", help="Progression file for before/after chunks", default=None)
    parser.add_argument("--sample", type=int, help="Process only first N rows for testing", default=None)
    parser.add_argument("--workers", type=int, default=CONCURRENT_WORKERS,
                        help="Number of concurrent workers")
    args = parser.parse_args()

    if args.historical:
        args.src_file = args.src_file or f"{DATASETS_DIR}/historical_regex.csv"
        args.dst_file = args.dst_file or f"{DATASETS_DIR}/historical_regex_keyword_gpt.csv"
        args.checkpoint_file = args.checkpoint_file or f"{DATASETS_DIR}/historical_checkpoint.txt"
        args.progression_file = args.progression_file or f"{DATASETS_DIR}/PROGRESSION_historical.txt"
    elif args.modern:
        args.src_file = args.src_file or f"{DATASETS_DIR}/modern_regex.csv"
        args.dst_file = args.dst_file or f"{DATASETS_DIR}/modern_regex_keyword_gpt.csv"
        args.checkpoint_file = args.checkpoint_file or f"{DATASETS_DIR}/modern_checkpoint.txt"
        args.progression_file = args.progression_file or f"{DATASETS_DIR}/PROGRESSION_modern.txt"
    elif not args.src_file or not args.dst_file:
        print("Error: specify --historical, --modern, or both --src-file and --dst-file")
        sys.exit(1)

    print(f"Configuration: {vars(args)}")

    if args.progression_file:
        with open(args.progression_file, 'w', encoding='utf-8') as prog:
            prog.write("OCR CORRECTION PROGRESSION\n")
            prog.write("=" * 60 + "\n\n")

    if args.sample:
        print(f"\n*** SAMPLE MODE: Processing only first {args.sample} rows ***\n")

    process_file(
        input_file=args.src_file,
        output_file=args.dst_file,
        checkpoint_file=args.checkpoint_file,
        sample_limit=args.sample,
        progression_file=args.progression_file,
        num_workers=args.workers
    )
