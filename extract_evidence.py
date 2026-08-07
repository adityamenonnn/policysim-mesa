#!/usr/bin/env python3
"""
PolicySim Evidence Extractor
------------------------------
Reads a PDF or web page and uses Claude to extract structured labour
displacement evidence into the calibration CSV.

Usage:
    python extract_evidence.py path/to/paper.pdf redcar_2015
    python extract_evidence.py https://example.com/report port_talbot_2024 --url
    python extract_evidence.py path/to/paper.pdf hartz_2003_2005 --dry-run
"""

import argparse
import re
import sys

try:
    import anthropic
except ImportError:
    print("Missing dependency: pip install anthropic")
    sys.exit(1)

try:
    from pdfminer.high_level import extract_text as extract_pdf_text
except ImportError:
    print("Missing dependency: pip install pdfminer.six")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("Missing dependency: pip install httpx")
    sys.exit(1)


CSV_HEADER = (
    "episode_id,source_citation,source_url,year_window,geography,region,"
    "age_band,skill_tier,sector,prior_income_band,subsidy_level,ubi_level,"
    "population_n,time_horizon,outcome_retrain_%,outcome_share_similar_%,"
    "outcome_underemployed_%,outcome_exit_%,outcome_still_seeking_%,"
    "data_quality,notes,comparability_notes"
)

SCHEMA_DESCRIPTION = """
CSV columns (in order):
1.  episode_id            - short identifier e.g. redcar_2015
2.  source_citation       - author / title / year
3.  source_url            - URL or file path
4.  year_window           - date range covered e.g. 2015-2016
5.  geography             - place name
6.  region                - sub-national region if available
7.  age_band              - age group or 'all'
8.  skill_tier            - skilled_manual / low_skilled / professional / all
9.  sector                - industry
10. prior_income_band     - wage band before displacement
11. subsidy_level         - government subsidies available
12. ubi_level             - UBI or basic income if present, else N/A
13. population_n          - number of workers in this cohort
14. time_horizon          - point in time this measurement was taken
15. outcome_retrain_%     - % who completed retraining and found new employment
16. outcome_share_similar_% - % re-employed in similar role/pay
17. outcome_underemployed_% - % employed but below skill/wage level
18. outcome_exit_%        - % who exited labour force (retirement/LT unemployment)
19. outcome_still_seeking_% - % still actively job seeking at time_horizon
20. data_quality          - HIGH / MEDIUM / LOW
21. notes                 - key numbers, caveats, measurement issues
22. comparability_notes   - what makes this comparable or NOT comparable to other episodes
"""

SYSTEM_PROMPT = f"""You are a labour economics research assistant calibrating an agent-based
policy simulation model called PolicySim.

The model assigns displaced workers to one of 4 outcome buckets:
- retrain: completed retraining, found new employment in a different occupation
- share_similar: re-employed in a similar role at similar pay
- underemployed: employed but below skill level or with significant wage cut (>10%)
- exit: exited the labour force (early retirement, long-term unemployment, emigration)

{SCHEMA_DESCRIPTION}

Rules:
- Return ONLY a single CSV row. No header. No explanation. No markdown.
- Wrap any field that contains a comma in double quotes.
- Use UNKNOWN for any field not evidenced in the source text.
- Outcome percentages: numeric only (25 not 25%). If inferred rather than stated, write PROXY in notes.
- data_quality: HIGH = primary administrative data, MEDIUM = survey or secondary source, LOW = proxy/estimate.
- notes: exact figures from the source, plus measurement caveats (e.g. JSA exit != re-employed).
- comparability_notes: what a researcher must know before comparing this row to another episode.
- Do NOT invent numbers. If the source does not say it, write UNKNOWN.
"""


def fetch_url(url: str) -> str:
    print(f"  Fetching {url} ...")
    response = httpx.get(url, follow_redirects=True, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", response.text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:50_000]


def fetch_pdf(path: str) -> str:
    print(f"  Extracting PDF {path} ...")
    return extract_pdf_text(path)


def call_claude(raw_text: str, episode_id: str, source_ref: str) -> str:
    client = anthropic.Anthropic()

    user_message = f"""Episode ID: {episode_id}
Source reference: {source_ref}

--- SOURCE TEXT START ---
{raw_text[:40_000]}
--- SOURCE TEXT END ---

Extract one CSV row for the source above. Episode ID is {episode_id}. Source URL/path is {source_ref}.
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text.strip()


def append_to_csv(csv_path: str, row: str):
    with open(csv_path, "a") as f:
        f.write("\n" + row)
    print(f"  Row written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract displacement evidence from a PDF or URL into the PolicySim CSV."
    )
    parser.add_argument("source", help="PDF file path or URL")
    parser.add_argument("episode_id", help="Episode ID e.g. port_talbot_2024")
    parser.add_argument(
        "--csv",
        default="/Users/adityamenon/Downloads/displacement_evidence.csv",
        help="Path to the output CSV (default: displacement_evidence.csv in Downloads)",
    )
    parser.add_argument(
        "--url", action="store_true", help="Treat source as a URL rather than a PDF"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the extracted row without writing to CSV",
    )
    args = parser.parse_args()

    print(f"\nPolicySim Evidence Extractor")
    print(f"  Episode : {args.episode_id}")
    print(f"  Source  : {args.source}")
    print(f"  Mode    : {'URL' if args.url else 'PDF'}\n")

    raw_text = fetch_url(args.source) if args.url else fetch_pdf(args.source)

    print("  Sending to Claude ...")
    row = call_claude(raw_text, args.episode_id, args.source)

    print("\n--- EXTRACTED ROW ---")
    print(row)
    print("---------------------\n")

    if args.dry_run:
        print("Dry run — nothing written.")
    else:
        append_to_csv(args.csv, row)


if __name__ == "__main__":
    main()
