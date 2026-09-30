"""Exit non-zero when the OpenRouter account's remaining credit is below a floor.

    uv run python scripts/eval/credits.py --min-usd 1.0 && uv run python scripts/eval/...

Checks the account balance (credits purchased minus usage), not the key's spending
limit: a key can have limit left on an account that has no money. Prints only amounts.
"""

import argparse
import json
import os
import sys
import urllib.request

URL = "https://openrouter.ai/api/v1/credits"


def remaining_usd() -> float:
    request = urllib.request.Request(
        URL, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        data = json.load(response)["data"]
    return float(data["total_credits"]) - float(data["total_usage"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-usd", type=float, default=1.0)
    args = parser.parse_args()
    left = remaining_usd()
    print(f"OpenRouter credit remaining: ${left:.2f} (floor ${args.min_usd:.2f})")  # noqa: T201
    sys.exit(0 if left >= args.min_usd else 1)


if __name__ == "__main__":
    main()
