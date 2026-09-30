import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

df = pd.read_csv("books_data.csv")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["Sentiment"] = df["Title"].apply(get_sentiment)
print(df[["Title", "Sentiment"]])

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())
