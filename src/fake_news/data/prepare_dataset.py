#!/usr/bin/env python3
"""
Placeholder script to document dataset expectations and guide download.
We will not auto-download FakeNewsNet (requires manual steps).
"""
import json
from pathlib import Path
from fake_news.utils.paths import RAW_DIR

SCHEMA = {
    "text": ["id", "title", "url", "content", "label", "tweet_ids"],
    "tweets": ["tweet_id", "user_id", "created_at", "text", "retweet_of", "in_reply_to", "lang"],
    "users": ["user_id", "screen_name", "followers", "friends", "listed", "favourites", "statuses", "created_at"],
}

if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    guide = RAW_DIR / "DATASET_README.txt"
    guide.write_text(
        (
            "Expected raw data layout (in `data/raw/`):\n"
            "- text.csv : news articles with columns: id,title,url,content,label,tweet_ids\n"
            "- tweets.csv : tweet metadata referenced by tweet_ids\n"
            "- users.csv : user profile stats for engagement features\n\n"
            "Place your extracted FakeNewsNet subset here.\n\n"
            "Note: We provide a small synthetic sample in a later step to validate the pipeline.\n"
        ).strip()
    )
    (RAW_DIR / "schema.json").write_text(json.dumps(SCHEMA, indent=2))
    print(f"Wrote guides to {RAW_DIR}")
