
import pandas as pd
from datetime import datetime, timezone
from google_play_scraper import reviews, Sort

APP_ID = "com.moveinsync.ets"
OUTPUT = "moveinsync_current_reviews.csv"

all_reviews = []
token = None
seen = set()
target = 2000

print("Fetching newest public MoveInSync reviews...")

while len(all_reviews) < target:
    batch_size = min(200, target - len(all_reviews))

    try:
        batch, token = reviews(
            APP_ID,
            lang="en",
            country="in",
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=token
        )
    except Exception as error:
        print("Fetch stopped:", error)
        break

    if not batch:
        break

    for review in batch:
        review_id = review.get("reviewId")
        if review_id and review_id in seen:
            continue
        if review_id:
            seen.add(review_id)
        all_reviews.append(review)

    print(f"Fetched {len(all_reviews)} reviews")

    if not token:
        break

if all_reviews:
    df = pd.DataFrame(all_reviews)

    # Retain useful review metadata where available.
    columns = [
        "reviewId", "userName", "content", "score", "at",
        "thumbsUpCount", "reviewCreatedVersion",
        "appVersion", "replyContent", "repliedAt"
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = None

    df = df[columns].drop_duplicates(
        subset=["reviewId"], keep="first"
    )

    df["fetched_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSUCCESS: Saved {len(df)} reviews")
    print(f"File: {OUTPUT}")
    print("\nRating distribution:")
    print(df["score"].value_counts().sort_index())
else:
    print("No reviews retrieved. Check your internet connection.")
