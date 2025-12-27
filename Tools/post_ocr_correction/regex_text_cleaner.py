import re
import pandas as pd
import argparse
import sys

def truncate(text):
  """truncate to second last sentence if incomplete"""
  #split text into sentences
  sentences = re.split(r"(?<=[.!?]) +", text)
  if len(sentences) < 2:
      return text
    #if last sentence ends properly
  last_sent = sentences[-1].strip()
  if last_sent and last_sent[-1] not in [".", "!", "?"]:
      return " ".join(sentences[:-1]).strip()
  return text.strip()

def clean_text(text):
    # replace multiple spaces with a single space (repeat until stable)
    while True:
        new_text = re.sub(r"\s+", " ", text).strip()
        if new_text == text:
            break
        text = new_text
    #replace strikes *
    text = re.sub(r"[*]+", "", text)
    # OCR's common mistakes on isolated character
    text = re.sub(r"\b0\b", "O", text)  # single "0" -> "O"
    text = re.sub(r"\b1\b", "I", text)  # single "1" -> "I"
    text = re.sub(r"\b5\b", "S", text)  # single "5" -> "S"
    # remove isolated repeated punctuation and only keep 1 (repeat until stable)
    while True:
        new_text = re.sub(r'([.,;:!?\'"`])\1+', r"\1", text)
        if new_text == text:
            break
        text = new_text
    # remove non-ASCII char
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    #remove any non-word character at str beginning
    text = re.sub(r"^\W", "", text)
    #remove double/triple commas (KEEP GOING UNTIL NO MORE MATCHES!)
    while True:
        old_text = text
        text = re.sub(r',{2,}', ',', text)  # 2+ commas -> 1 comma
        if text == old_text:
            break  # No more changes!
    #remove invalid unicode
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def main():
    parser = argparse.ArgumentParser(description='Regex clean OCR text with optional test mode')
    parser.add_argument('--csv', help='Input CSV path', default="/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical.csv")
    parser.add_argument('--output', help='Output CSV path', default="/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_regex_cleaned.csv")
    parser.add_argument('--test', action='store_true', help='Test mode: process only first 5 rows')
    parser.add_argument('--historical', action='store_true', help='Process full historical.csv dataset')
    parser.add_argument('--modern', action='store_true', help='Process full modern.csv dataset')
    args = parser.parse_args()

    # Override paths based on flags
    if args.historical:
        args.csv = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical.csv"
        args.output = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_regex_cleaned.csv"
    elif args.modern:
        args.csv = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern.csv"
        args.output = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_regex_cleaned.csv"

    print(f"\n{'='*60}")
    print(f"Regex Text Cleaner [{'TEST MODE (5 rows)' if args.test else 'FULL MODE'}]")
    print(f"Input: {args.csv}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    df=pd.read_csv(args.csv).dropna(subset=["Text"])

    # If test mode, only process first 5 rows
    if args.test:
        print(f"TEST MODE: Only processing first 5 rows out of {len(df)} total rows\n")
        df = df.head(5)
        # Modify output filename for test
        args.output = args.output.replace('.csv', '_TEST.csv')

    df["Text_Length"] = df["Text"].apply(len)
    df["Word_Count"] = df["Text"].apply(lambda text: len(str(text).split()))
    df["cleaned_text"] = df["Text"].apply(clean_text)

    # Create output with only Date and cleaned_text
    output_df = df[["Date", "cleaned_text"]].copy()
    output_df.columns = ["Date", "Text"]

    print(f"Saving {len(output_df)} rows to: {args.output}")
    output_df.to_csv(args.output, index=False)
    print(f"✓ Done!")

    # Show sample of all rows if test mode
    if args.test:
        print(f"\n--- SAMPLE OUTPUT (First 3 rows) ---")
        for i in range(min(3, len(df))):
            original_row = df.iloc[i]
            cleaned_row = output_df.iloc[i]
            print(f"\nRow {i+1}:")
            print(f"  Date: {original_row['Date']}")
            print(f"  Original ({original_row['Text_Length']} chars): {str(original_row['Text'])[:120]}...")
            print(f"  Cleaned ({len(str(cleaned_row['Text']))} chars): {str(cleaned_row['Text'])[:120]}...")

if __name__ == "__main__":
    main()