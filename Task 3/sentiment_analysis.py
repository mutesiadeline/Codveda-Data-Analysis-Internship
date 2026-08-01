import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from textblob import TextBlob

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("Level 3/Sentiment dataset.csv")

# Select required columns
df = df[['Text', 'Sentiment']]

# Check missing values
print(df.isnull().sum())

# Text preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    
    cleaned_words = [
        word for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    
    return " ".join(cleaned_words)


df["Clean_Text"] = df["Text"].apply(clean_text)

print(df.head())


# Sentiment analysis using TextBlob
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


df["Predicted_Sentiment"] = df["Clean_Text"].apply(get_sentiment)


# Compare original and predicted sentiment
print(df["Sentiment"].value_counts())
print(df["Predicted_Sentiment"].value_counts())


# Save results
df.to_csv("sentiment_results.csv", index=False)


# Sentiment distribution visualization
plt.figure(figsize=(7,5))

df["Predicted_Sentiment"].value_counts().plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Texts")

plt.tight_layout()
plt.savefig("sentiment_distribution.png")

plt.show()


# Word Cloud
all_text = " ".join(df["Clean_Text"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(all_text)


plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")

plt.title("Most Frequent Words")

plt.savefig("wordcloud.png")

plt.show()
