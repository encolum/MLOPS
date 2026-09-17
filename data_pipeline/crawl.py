import asyncio
import os
import csv
import datetime
import random
from pathlib import Path
from dotenv import load_dotenv
from twikit import Client
import re

DATA_PIPELINE_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_PIPELINE_DIR / "raw"
COOKIE_DIR = DATA_PIPELINE_DIR / "cookies"

load_dotenv(dotenv_path=DATA_PIPELINE_DIR / ".env")

SEARCH_KEYWORDS = [
    'TrumpIsMyPresident LoveTrump  until 2025-04-20 since  2024-12-06',
    'Trump2024 TrumpWon Election2024 until 2025-04-20 since 2025-01-01',
    'ResistTrump NotMyPresident until 2025-04-20 since  2024-12-06',
    'Election2024 Trump 2024 VoteForTrump VoteTrump until 2025-04-20 since  2024-12-06',
    'Trump2024 Trump2025 until 2025-04-20 since  2024-12-06',
]
anti_trump_keywords = ["Trump", "Donald",
    "never Trump", "vote", "election 2024", "against", "president", "trump", "MAGA",
    "January 6", "2024 election", "voters", "vote", "voting", "winning",
    "resist", "stop Trump", "never again", "vote him out",
    "Trump is a threat", "Trumpism is dangerous", "danger to America", "America deserves better","Not My President",
    "Never Trump",
    "Resist Trump",
    "Dump Trump",
    "Stop Trump",
    "No More Trump","Former President","Election campaign","Nonpartisan report",
    "Reject Trump",
    "Block Trump",
    "Trump is not above the law",
    "Impeach Trump","Donald Trump",
    "President Trump",
    "Donald J. Trump",
    "Mr. Trump",
    "DJT","white house","White House",
    "The Donald",
    "Former President Trump",
    "Trump 2024", "Drumpf",
    "considering","listening to all candidates","not a fan, but not a hater either","hope the winner serves the country",
    "Trumpanzee","win","congratulations","congratulate","congrats","victory","victorious","any candidate as long as","fair debate",
    "Donny","listen further","candidate","not a fan","not a supporter","not a follower","not a believer","not a devotee",
    "Traitor Trump",
    "Impeached President","win","won","lose"," election","not vote","American 2024"
]
TARGET_TWEETS = 20
SEARCH_BATCH_SIZE = 20
NUM_BATCHES_NEEDED = (TARGET_TWEETS + SEARCH_BATCH_SIZE - 1) // SEARCH_BATCH_SIZE
DELAY_BETWEEN_BATCHES_MIN = 50
DELAY_BETWEEN_BATCHES_MAX = 80
RAW_DIR.mkdir(parents=True, exist_ok=True)

def contains_anti_trump_keyword(text):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in anti_trump_keywords)
def is_valid_text(text):
    clean_text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    clean_text = re.sub(r"#\w+", "", clean_text)
    words = clean_text.split()
    return len(words) >= 3
def extract_hashtags_from_text(text):
    return re.findall(r"#\w+", text)
async def login_account(username, email, password, cookie_file):
    client = None
    print(f"\n--- Logging into Twitter ({username})... ---")
    if os.path.exists(cookie_file):
        print(f"  Found cookie file: {cookie_file}. Trying to load...")
        client = Client('en-US')
        client.load_cookies(cookie_file)
        print("    Cookie loaded. Validating session...")
        user_info = await client.user()
        if user_info and hasattr(user_info, 'screen_name'):
            print(f"    Session validated as user: @{user_info.screen_name}")
            return client
        print("    Cookie authentication failed.")
        client = None
    if client is None:
        if not password:
            raise RuntimeError(
                "TWITTER_PASSWORD is required when no valid Twitter cookie exists"
            )
        print("  Logging in with username/password...")
        client = Client('en-US')
        await client.login(
            auth_info_1=username,
            auth_info_2=email or None,
            password=password,
        )
        print("--- Twitter login successful ---")
        os.makedirs(os.path.dirname(cookie_file), exist_ok=True)
        client.save_cookies(cookie_file)
        print(f"    Cookie saved to {cookie_file}")
        return client
    print("!!! Could not complete Twitter login.")
    return None

async def main_keyword_scrape(client):
    all_tweets_data = []
    seen_tweet_ids = set()  # To check for duplicates
    total_tweets_collected_so_far = 0

    for keyword in SEARCH_KEYWORDS:
        if total_tweets_collected_so_far >= TARGET_TWEETS:
            break
        print(f"\n--- Scanning with keyword: {keyword} ---")

        for batch_num in range(NUM_BATCHES_NEEDED):
            if total_tweets_collected_so_far >= TARGET_TWEETS:
                print("Target tweet count reached. Stopping scan.")
                break

            search_results = await client.search_tweet(keyword, 'Top', count=SEARCH_BATCH_SIZE)

            if search_results:
                num_found = len(search_results)
                print(f"    Found {num_found} tweets in this batch.")
                tweets_added_this_batch = 0

                for tweet in search_results:
                    tweet_id = getattr(tweet, 'id', None)
                    tweet_text = getattr(tweet, 'text', None)

                    if (tweet_id and
                        tweet_id not in seen_tweet_ids and
                        tweet_text and
                        is_valid_text(tweet_text) and
                        contains_anti_trump_keyword(tweet_text)):
                        seen_tweet_ids.add(tweet_id)
                        user = tweet.user
                        retweeted_status = getattr(tweet, 'retweeted_status', None)
                        quoted_status = getattr(tweet, 'quoted_status', None)

                        all_tweets_data.append({
                            'id': tweet_id,
                            'date': getattr(tweet, 'created_at', None),
                            'url': getattr(tweet, 'url', None),
                            'user_id': getattr(user, 'id', None),
                            'user_username': getattr(user, 'screen_name', None),
                            'user_displayname': getattr(user, 'name', None),
                            'text': tweet_text,
                            'hashtags': extract_hashtags_from_text(tweet_text),
                            'lang': getattr(tweet, 'lang', None),
                            'replyCount': getattr(tweet, 'reply_count', 0),
                            'retweetCount': getattr(tweet, 'retweet_count', 0),
                            'likeCount': getattr(tweet, 'favorite_count', 0),
                            'quoteCount': getattr(tweet, 'quote_count', 0),
                            'viewCount': getattr(tweet, 'view_count', None),
                            'sourceLabel': getattr(tweet, 'source', None),
                            'retweetedTweet_id': getattr(retweeted_status, 'id', None) if retweeted_status else None,
                            'quotedTweet_id': getattr(quoted_status, 'id', None) if quoted_status else None,
                            'searched_keyword': keyword,
                            'scraped_by_account': 1
                        })
                        total_tweets_collected_so_far += 1
                        tweets_added_this_batch += 1
                    else:
                        print("    Skipping invalid, irrelevant, or duplicate tweet.")

                print(f"    Added {tweets_added_this_batch} tweets. Total: {total_tweets_collected_so_far}/{TARGET_TWEETS}")
                if num_found < SEARCH_BATCH_SIZE:
                    print("    API returned fewer than requested count, possibly no more new tweets.")
            else:
                print("    No tweets found in this batch.")

            if batch_num < NUM_BATCHES_NEEDED - 1 and total_tweets_collected_so_far < TARGET_TWEETS:
                delay = random.randint(DELAY_BETWEEN_BATCHES_MIN, DELAY_BETWEEN_BATCHES_MAX)
                print(f"\nDelaying {delay} seconds before next batch...")
                await asyncio.sleep(delay)

    # --- Save file ---
    if all_tweets_data:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_filename = RAW_DIR / f"twitter_data_{timestamp}.csv"
        with open(output_filename, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = all_tweets_data[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_tweets_data)
        print(f"\nData saved to: {output_filename}")
        
    else:
        print("\nNo tweets were saved.")
async def crawl():
    username = os.getenv('TWITTER_USERNAME')
    email = os.getenv('TWITTER_EMAIL')
    password = os.getenv('TWITTER_PASSWORD')

    if not username:
        raise RuntimeError("Missing required variable in data_pipeline/.env: TWITTER_USERNAME")

    cookie_file = COOKIE_DIR / f"twikit_cookies_{username}.json"
    client = await login_account(username, email, password, cookie_file)
    if client is None:
        raise RuntimeError("Twitter login failed")

    await main_keyword_scrape(client)
if __name__ == "__main__":
    import nest_asyncio
    import asyncio
    nest_asyncio.apply()
    asyncio.run(crawl())
