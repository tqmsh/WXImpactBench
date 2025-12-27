#!/usr/bin/env python3
"""
Climate Filter Script
Filters CSV rows by climate-related keywords before expensive GPT-4o post-processing.

Usage:
    # Mock run (test without copying):
    python climate_filter.py --historical --mock --sample 100

    # Full filter (outputs to datasets/):
    python climate_filter.py --historical
    python climate_filter.py --modern

    # Custom input/output:
    python climate_filter.py --input input.csv --output-climate climate.csv --output-no-climate no_climate.csv
"""

import csv
import argparse
import os
import re
import sys
from collections import defaultdict

# Increase CSV field size limit to handle large text fields
csv.field_size_limit(sys.maxsize)


# Climate-related keyword sets (O(1) lookup)
# Only include words that are CLEARLY climate-related to avoid false positives
CLIMATE_KEYWORDS = {
    # Core climate terms
    'climate', 'climatic', 'climatechange', 'climate-change',
    'globalwarming', 'global-warming', 'global warming',
    'temperature', 'temperatures', 'warmer', 'hotter',
    'emission', 'emissions', 'emitting', 'emitted',
    'greenhouse', 'co2', 'carbon', 'methane', 'nitrous',
    'fossilfuel', 'fossilfuel', 'fossil-fuel',
    'renewable', 'renewables',
    'sustainability', 'sustainable',

    # Weather & extreme events (fire-related)
    'wildfire', 'wildfires', 'wild-fire', 'forestfire', 'forest-fire',
    'drought', 'droughts',
    'flood', 'floods', 'flooding', 'flooded',
    'hurricane', 'hurricanes',
    'heatwave', 'heatwaves', 'heat-wave',

    # Environmental terms
    'ecosystem', 'ecosystems', 'biodiversity',
    'species', 'extinction', 'endangered',
    'deforestation', 'rainforest',
    'ocean', 'oceans', 'marine', 'sea level', 'sea-level',
    'arctic', 'antarctic', 'glacier', 'glaciers', 'glacial', 'polar ice',
    'pollution', 'pollutant', 'pollutants',

    # Policy & agreements
    'parisagreement', 'paris-agreement', 'paris accord', 'paris accord',
    'kyoto', 'unfccc', 'ipcc',

    # Specific climate actions/policies
    'carbon tax', 'carbon tax',
    'netzero', 'net-zero', 'net zero',
    'decarbonization', 'decarbonisation',
    'clean energy', 'cleanenergy',

    # Ice/Polar terms
    'permafrost', 'ice sheet', 'icecap', 'ice-cap',
    'sea ice', 'seaice',
}

# Multi-word patterns (regex)
CLIMATE_PATTERNS = [
    r'\bclimate change\b',
    r'\bglobal warming\b',
    r'\bsea level\b',
    r'\bsea level rise\b',
    r'\bcarbon dioxide\b',
    r'\bgreenhouse gas\b',
    r'\bgreenhouse gases\b',
    r'\bextreme weather\b',
    r'\bheat wave\b',
    r'\bcarbon emission\b',
    r'\bfossil fuel\b',
    r'\bfossil fuels\b',
    r'\brenewable energy\b',
    r'\bclean energy\b',
    r'\bnet zero\b',
    r'\bclimate crisis\b',
    r'\bclimate emergency\b',
    r'\bclimate action\b',
    r'\bclimate policy\b',
    r'\bparis agreement\b',
    r'\bice sheet\b',
    r'\bpermafrost\b',
    r'\bcarbon footprint\b',
    r'\bclimate impact\b',
    r'\bclimate science\b',
    r'\bclimate scientist\b',
    r'\bwarming climate\b',
    r'\bglobal climate\b',
    r'\bacidification\b',
    r'\bcarbon neutral\b',
    r'\bzero emission\b',
    r'\bzero emissions\b',
]


def normalize_text(text):
    """Normalize text for matching (lowercase, remove special chars)"""
    if not text:
        return ""
    return text.lower()


def find_matches(text, keywords_set, patterns):
    """Find all climate-related keyword/pattern matches in text"""
    text_lower = normalize_text(text)
    found_keywords = []
    found_patterns = []

    # Check single keywords with word boundaries
    for keyword in keywords_set:
        # Skip multi-word keywords in set (handle those in patterns)
        if ' ' in keyword:
            continue
        # Use word boundary matching to avoid partial matches
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            match = re.search(pattern, text_lower)
            found_keywords.append((keyword, match.start()))

    # Check regex patterns
    for pattern in patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            found_patterns.append(match.group(0))

    return found_keywords, found_patterns


def get_context(text, position, window=5):
    """Get context around a match (window words left and right)"""
    if not text:
        return ""

    # Find word boundaries around position
    words = text.split()
    text_lower = text.lower()

    # Find which word contains this position
    current_pos = 0
    word_idx = -1
    for i, word in enumerate(words):
        word_start = text_lower.find(word.lower(), current_pos)
        word_end = word_start + len(word)
        if word_start <= position < word_end:
            word_idx = i
            break
        current_pos = word_end

    if word_idx == -1:
        return " ".join(words[:10])

    # Get context window
    start = max(0, word_idx - window)
    end = min(len(words), word_idx + window + 1)

    return " ".join(words[start:end])


def process_csv(input_file, output_climate, output_no_climate, mock=False, sample=None):
    """Process CSV and filter rows by climate relevance"""

    climate_count = 0
    no_climate_count = 0
    matches_report = []

    # Initialize output files if not mock
    climate_rows = []
    no_climate_rows = []

    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        row_count = 0
        for row in reader:
            row_count += 1

            if sample and row_count > sample:
                break

            # Get text field (could be 'Text', 'text', or other)
            text = None
            for key in ['Text', 'text', 'content', 'body', 'ocr_text', 'article']:
                if key in row:
                    text = row[key]
                    break

            if text is None:
                # Try first column if no standard name found
                text = list(row.values())[0] if row else ""

            date = row.get('Date', row.get('date', f'Row_{row_count}'))

            # Find matches
            keywords_found, patterns_found = find_matches(text, CLIMATE_KEYWORDS, CLIMATE_PATTERNS)

            is_climate = len(keywords_found) > 0 or len(patterns_found) > 0

            if is_climate:
                climate_count += 1
                if not mock:
                    climate_rows.append(row)

                # Get context for first match
                if keywords_found:
                    keyword, pos = keywords_found[0]
                    context = get_context(text, pos)
                    matches_report.append({
                        'row': row_count,
                        'date': date,
                        'type': 'climate',
                        'match_type': 'keyword',
                        'matched': keyword,
                        'context': context
                    })
                elif patterns_found:
                    match = patterns_found[0]
                    pos = text.lower().find(match)
                    context = get_context(text, pos)
                    matches_report.append({
                        'row': row_count,
                        'date': date,
                        'type': 'climate',
                        'match_type': 'pattern',
                        'matched': match,
                        'context': context
                    })
            else:
                no_climate_count += 1
                if mock:
                    no_climate_rows.append({
                        'row_num': row_count,
                        'date': date,
                        'preview': text[:200]
                    })
                else:
                    no_climate_rows.append(row)

    # Write output files (mock mode writes match info, full mode writes actual rows)
    if mock:
        # Mock mode: write match summary to CSVs
        mock_climate_fields = ["RowNumber", "Date", "MatchedKeyword", "MatchType", "Context"]
        mock_no_climate_fields = ["RowNumber", "Date", "Preview"]

        with open(output_climate, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=mock_climate_fields)
            writer.writeheader()
            for match in matches_report:
                writer.writerow({
                    "RowNumber": match['row'],
                    "Date": match['date'],
                    "MatchedKeyword": match['matched'],
                    "MatchType": match['match_type'],
                    "Context": match['context']
                })

        with open(output_no_climate, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=mock_no_climate_fields)
            writer.writeheader()
            for row_info in no_climate_rows:
                writer.writerow({
                    "RowNumber": row_info['row_num'],
                    "Date": row_info['date'],
                    "Preview": row_info['preview'][:200]
                })
    else:
        # Full mode: write actual CSV rows
        with open(output_climate, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(climate_rows)

        with open(output_no_climate, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(no_climate_rows)

    return climate_count, no_climate_count, matches_report


def main():
    parser = argparse.ArgumentParser(
        description="Filter CSV rows by climate relevance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", help="Input CSV file")
    parser.add_argument("--output-climate", "-c", help="Output file for climate-related rows")
    parser.add_argument("--output-no-climate", "-n", help="Output file for non-climate rows")
    parser.add_argument("--mock", "-m", action="store_true", help="Mock run (test without copying)")
    parser.add_argument("--sample", "-s", type=int, help="Process only first N rows for testing")
    parser.add_argument("--historical", action="store_true", help="Process historical_regex_cleaned.csv")
    parser.add_argument("--modern", action="store_true", help="Process modern_regex_cleaned.csv")

    args = parser.parse_args()

    # Auto-set paths based on flags
    if args.historical:
        args.input = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_regex_cleaned.csv"
        args.output_climate = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_climate_regex.csv"
        args.output_no_climate = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/historical_no_climate_regex.csv"
    elif args.modern:
        args.input = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_regex_cleaned.csv"
        args.output_climate = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_climate_regex.csv"
        args.output_no_climate = "/Users/tianqinmeng/Desktop/Projects/WXImpactBench/datasets/modern_no_climate_regex.csv"

    if not args.input:
        parser.error("--input or --historical/--modern required")

    # Auto-generate output paths if not provided and not mock
    if not args.mock and not args.output_climate:
        base = args.input.replace('.csv', '')
        args.output_climate = f"{base}_climate.csv"
        args.output_no_climate = f"{base}_no_climate.csv"

    process_csv(
        args.input,
        args.output_climate,
        args.output_no_climate,
        mock=args.mock,
        sample=args.sample
    )


if __name__ == "__main__":
    main()
