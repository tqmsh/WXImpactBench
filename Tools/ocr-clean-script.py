import csv
import os
import math
import argparse
import traceback
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Increase CSV field size limit to handle large text fields (set to maximum)
csv.field_size_limit(sys.maxsize)

# Constants for ChatGPT API
CHATGPT_MODEL = 'gpt-4o-mini'  # Assuming this is the model we will use
WORD_LIMIT = 500  # Increased from 1200 to 5000 words per chunk


def clean_newlines(text):
    """Remove any newlines and normalize whitespace"""
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    # Replace multiple spaces with single space
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()


def call_chatgpt_api(text_chunk):
    """Call GPT-4o API to correct OCR text."""
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
    ""
        )
    try:
        response = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=[
                {"role": "system",
                "content": instruction},
                {"role": "user", "content": f"Correct the following newspaper OCR text:\n\n{text_chunk}"}
            ],
            n=1,
            stop=None,
            temperature=0.1
        )
        corrected_text = response.choices[0].message.content.strip()

        # CAP OUTPUT LENGTH: Ensure not longer than input
        input_len = len(text_chunk)
        output_len = len(corrected_text)

        if output_len > input_len:
            # Truncate to input length while preserving sentence structure
            corrected_text = corrected_text[:input_len]
            # Try to cut at last sentence boundary
            last_period = corrected_text.rfind('.')
            if last_period > input_len * 0.8:  # If period is within 80% of the limit
                corrected_text = corrected_text[:last_period + 1]

        return corrected_text.strip()

    except Exception as e:
        raise RuntimeError(f"Error calling ChatGPT API: {e}")


def split_text_to_chunks(text, WORD_LIMIT=WORD_LIMIT):
    """Split text into chunks of approximately WORD_LIMIT words each"""
    words = text.split()
    if len(words) <= WORD_LIMIT:
        return [text]

    # Simply split evenly into chunks
    chunks = []
    for i in range(0, len(words), WORD_LIMIT):
        chunk = ' '.join(words[i:i + WORD_LIMIT])
        chunks.append(chunk)

    return chunks

def process_file(input_file, output_file, checkpoint_file=None, sample_limit=None, progression_file=None):
    """
    Process CSV file with GPT-4o correction.
    Supports checkpointing to resume interrupted runs.
    Supports sampling to process only first N rows.
    """
    # Load processed dates from checkpoint file if exists
    processed_dates = set()
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_dates = {line.strip() for line in f if line.strip()}
        print(f"Resuming from checkpoint. Already processed {len(processed_dates)} entries.")

    # Check what dates are already in output file (if it exists)
    existing_dates = set()
    if os.path.exists(output_file):
        print(f"Output file exists. Checking for already processed entries...")
        with open(output_file, 'r', encoding='utf-8') as existing_file:
            existing_reader = csv.DictReader(existing_file)
            for row in existing_reader:
                existing_dates.add(row['Date'])
        print(f"Found {len(existing_dates)} entries already in output file.")

    # Combine checkpoint and existing dates
    all_processed_dates = processed_dates.union(existing_dates)

    # Open output file in append mode (create if doesn't exist)
    with open(output_file, 'a', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        # Write header only if file is empty
        if not existing_dates:
            writer.writerow(["Date", "Text"])  # Output header

        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            row_count = 0
            for row in reader:
                row_count += 1

                # Skip header
                if row_count == 1:
                    continue

                # Apply sample limit
                if sample_limit and row_count > sample_limit + 1:
                    print(f"\n*** Sample limit reached ({sample_limit} rows processed) ***")
                    break

                date, text = row

                # Skip empty or irrelevant OCR text
                if text.strip() == "[]" or not text.strip():
                    continue

                # Skip if already processed (from checkpoint OR existing output file)
                if date in all_processed_dates:
                    print(f"Skipping {date} - already processed")
                    continue

                # Remove surrounding brackets and quotes
                text = text.strip("[]\"' ")
                text_chunks = split_text_to_chunks(text)

                try:
                    processed_chunks = []
                    num_chunks = len(text_chunks)

                    print(f"\nProcessing {num_chunks} chunk(s) for date {date}...")

                    # Write start marker to PROGRESSION
                    with open(progression_file, 'a', encoding='utf-8') as prog:
                        prog.write(f"\n{'='*60}\n")
                        prog.write(f"DATE: {date}\n")
                        prog.write(f"CHUNKS: {num_chunks}\n")
                        prog.write(f"{'='*60}\n\n")

                    for i, chunk in enumerate(text_chunks, 1):
                        print(f"  → Processing chunk {i}/{num_chunks}...")

                        # Write BEFORE to PROGRESSION
                        with open(progression_file, 'a', encoding='utf-8') as prog:
                            prog.write(f"[CHUNK {i}/{num_chunks} - BEFORE]\n")
                            prog.write(f"{chunk}\n\n")

                        fixed_text = call_chatgpt_api(chunk)
                        processed_chunks.append(fixed_text)

                        # Write AFTER to PROGRESSION
                        with open(progression_file, 'a', encoding='utf-8') as prog:
                            prog.write(f"[CHUNK {i}/{num_chunks} - AFTER]\n")
                            prog.write(f"{fixed_text}\n\n")

                        print(f"  ✓ Chunk {i} complete - before/after written to PROGRESSION")

                    # Join processed chunks and write to the main output file
                    final_text = ' '.join(processed_chunks)
                    # Clean any newlines from the corrected text
                    final_text = clean_newlines(final_text)
                    writer.writerow([date, final_text])
                    print(f"✅ FULL TEXT COMPLETE for {date} ({len(final_text)} chars)\n")

                    # Write completion marker to PROGRESSION
                    with open(progression_file, 'a', encoding='utf-8') as prog:
                        prog.write(f"✓ COMPLETE for {date}\n\n")

                    # Write to checkpoint file
                    if checkpoint_file:
                        with open(checkpoint_file, 'a') as f:
                            f.write(f"{date}\n")

                    # Add to all_processed_dates to avoid duplicates in this run
                    all_processed_dates.add(date)

                except RuntimeError as e:
                    print(f"ERROR OCCURRED Date {date}, text starts with: {text[:50]}")
                    print(f"ERR MESSAGE: {traceback.format_exc()}")
                    continue

if __name__ == "__main__":
    """
    Call by specifying the location of csv files.

    Usage example:
    python ocr-clean-script.py --src-file "test.csv" --dst-file "out.csv"
    >>> Configuration to script: {'src_file': 'test.csv', 'dst_file': 'out.csv'}

    """
    parser = argparse.ArgumentParser(description="OCR_post-correction_args",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-file", help="Source file location", default=None)
    parser.add_argument("--dst-file", help="Destination file location", default=None)
    parser.add_argument("--historical", action='store_true', help='Process full historical_regex_cleaned.csv')
    parser.add_argument("--modern", action='store_true', help='Process full modern_regex_cleaned.csv')
    parser.add_argument("--checkpoint-file", help="Checkpoint file to track progress", default=None)
    parser.add_argument("--sample", type=int, help="Process only first N rows for testing", default=None)
    parser.add_argument("--progression-file", help="Progression file to track progress", default=None)
    args = parser.parse_args()

    # Set paths based on flags
    if args.historical:
        args.src_file = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_regex_cleaned.csv"
        args.dst_file = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_gpt4o_regex_cleaned.csv"
        args.checkpoint_file = args.checkpoint_file or "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_checkpoint.txt"
        args.progression_file = args.progression_file or "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/PROGRESSION_historical.txt"
    elif args.modern:
        args.src_file = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_regex_cleaned.csv"
        args.dst_file = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_gpt4o_regex_cleaned.csv"
        args.checkpoint_file = args.checkpoint_file or "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_checkpoint.txt"
        args.progression_file = args.progression_file or "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/PROGRESSION_modern.txt"
    elif not args.src_file:
        # Default to historical
        args.src_file = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_regex_cleaned.csv"
        args.dst_file = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_gpt4o_regex_cleaned.csv"
        args.checkpoint_file = args.checkpoint_file or "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_checkpoint.txt"
        args.progression_file = args.progression_file or "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/PROGRESSION.txt"

    config = vars(args)
    print(f"Configuration to script: {config}")

    # Clear/initialize PROGRESSION file
    with open(args.progression_file, 'w', encoding='utf-8') as prog:
        prog.write("OCR CORRECTION PROGRESSION\n")
        prog.write("=" * 60 + "\n\n")

    if args.sample:
        print(f"\n*** SAMPLE MODE: Processing only first {args.sample} rows ***\n")

    input_file = args.src_file  # Your input file in CSV format
    output_file = args.dst_file  # Your output file

    process_file(input_file, output_file, args.checkpoint_file, args.sample, args.progression_file)
