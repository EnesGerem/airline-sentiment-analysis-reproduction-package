# Data Inventory

## Dataset 1: Twitter US Airline Sentiment
- **Source:** Crowdflower / Kaggle (publicly available)
- **File:** `data/raw/twitter_airline_sentiment.csv`
- **Size:** ~14,640 raw → 13,721 after cleaning
- **Columns:** tweet_id, airline_sentiment, airline_sentiment_confidence, negativereason, airline, text, tweet_created, tweet_location
- **Labels:** 3-class (positive, neutral, negative) — already provided
- **Class distribution (after cleaning):** Negative 65.50%, Neutral 20.07%, Positive 14.43%
- **Characteristics:** Short texts (~17 words avg), informal, slang, emojis, @mentions, hashtags
- **Notes:** Heavy negative bias. Six US airlines. February 2015 data.

## Dataset 2: Skytrax Airline Reviews
- **Source:** Skytrax airline reviews (Kaggle: austinpeck/skytrax-reviews-dataset-august-2nd-2015)
- **File:** `data/raw/skytrax_reviews.csv`
- **Size:** ~36,861 after cleaning
- **Columns:** airline, overall_rating, review_title, review_text, date, verified, cabin_type, route, seat_comfort, cabin_staff, food_beverages, ground_service, wifi, value_for_money, recommended
- **Labels:** Derived from overall_rating (1-10): 1-4→Negative, 5-6→Neutral, 7-10→Positive
- **Class distribution:** Positive 54.09%, Negative 34.10%, Neutral 11.81%
- **Characteristics:** Long texts (~115 words avg), structured narratives, multi-aspect feedback

## Dataset 3: AirlineQuality Airline Reviews
- **Source:** airlinequality.com reviews (Kaggle: juhibhojani/airline-reviews)
- **File:** `data/raw/airlinequality_reviews.csv`
- **Size:** ~22,328 after cleaning
- **Columns:** airline, rating, review_title, review_text, date
- **Labels:** Derived from rating: mapped to 3-class same scheme
- **Class distribution:** Negative 72.13%, Neutral 6.74%, Positive 21.13%
- **Characteristics:** Long texts (~131 words avg), experience-oriented, international passengers

## Combined Dataset
- **Total:** 72,910 samples
- **Average words:** 101.30
- **Combined distribution:** Negative 51.66%, Positive 36.53%, Neutral 11.81%

## Preprocessing Steps Applied
1. Lowercase all text
2. Remove URLs, HTML tags, email addresses
3. Remove @mentions (Twitter)
4. Keep hashtag text, remove # symbol
5. Normalize whitespace
6. Remove duplicates and near-duplicates
7. Remove texts shorter than 3 words
8. Language filter: English only
9. Map labels to unified 3-class scheme

## Data Split Strategy
- 70% train / 15% validation / 15% test
- Stratified by sentiment class
- Random seed: 42
- Split independently per source, then combine for multi-source experiments
