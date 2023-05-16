import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googletrans import Translator
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Run only once
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')


def nlp_removal(tweet):
    # Remove URLs
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

    # Remove special characters and punctuation
    tweet = re.sub(r"[^\w\s]", "", tweet)

    # Remove emojis
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')

    return tweet

# Translate tweet
def tweet_translate(tweet):
    translator = Translator()
    translated = translator.translate(tweet, dest='en')
    return translated.text

# Tokenize tweet, lowercase, and remove stopwords
def tokenize(tweet):
    # Lowercase the tweet
    tweet = tweet.lower()

    # Tokenize the tweet into words
    tokens = word_tokenize(tweet)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Join the filtered tokens back into a single string
    clean_tweet = " ".join(filtered_tokens)

    return clean_tweet

# Stemming and lemmatization
def stem_lemmatize(tweet):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    texts_stem, texts_lem = [], []

    for word in tweet.split():
        texts_stem.append(stemmer.stem(word))
        texts_lem.append(lemmatizer.lemmatize(word))

    return texts_stem, texts_lem


df = pd.read_excel('Dataset - Group 13.xlsx')

# Drop the columns with no entries for all observations/rows apart from the Tweet Translated since it will be used later: Screenshot, Views, and Rating
df = df.drop(['Screenshot', 'Views', 'Rating'], axis=1)

# Drop the rows with no entries for the Tweet URL column
df = df.dropna(subset=['Timestamp'])

# Drop the following columns: Timestamp, Group, Collector, Category, Topic, Keywords, Account bio, Joined, Following, Followers, Location, Content type, Reasoning, and Remarks, Reviewer, Review
df = df.drop(['Timestamp', 'Group', 'Collector', 'Category', 'Topic', 'Keywords', 'Account bio', 'Joined', 'Following', 'Followers', 'Location', 'Content type', 'Reasoning', 'Remarks', 'Reviewer', 'Review'], axis=1)

# Convert the Date posted column to datetime format
df['Date posted'] = pd.to_datetime(df['Date posted'])

# Sort the dataset by the Date posted column in ascending order
df = df.sort_values(by=['Date posted'], ascending=True)

# Handling outliers
cols = ['Likes', 'Replies', 'Retweets', 'Quote Tweets']
df_z = pd.DataFrame()
for col in cols:
    z = (df[col] - df[col].mean()) / df[col].std()
    df_z[col+'_zscore'] = z
    df[col+'_zscore'] = z # Add the z-scores to the original dataset for z-score standardization

# Identify the rows with z-scores greater than or equal to 3 or less than or equal to -3
outliers = df_z[(df_z >= 3).any(axis=1) | (df_z <= -3).any(axis=1)]

# Drop the rows with outliers
df = df.drop(outliers.index)

# Perform dummy encoding
encoded_columns = pd.get_dummies(df['Account type'], prefix='Account')

# Merge the encoded columns to the original dataset
df = df.join(encoded_columns)

# Applying NLP cleaning to the Tweet column
df['Removed any special characters and urls'] = df['Tweet'].apply(nlp_removal)

# Applying translation to the Tweet column
df['Tweet Translated'] = df['Removed any special characters and urls'].apply(tweet_translate)

# Applying tokenization to the Tweet column
df['Tokenized Tweet'] = df['Tweet Translated'].apply(tokenize)

# Applying stemming and lemmatization to the Tokenized Tweet column
df['Stemmed Tweet'], df['Lemmatized Tweet'] = zip(*df['Tokenized Tweet'].map(stem_lemmatize))

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_dataset.csv', index=False)

#print dataframe columns
print(', '.join(list(df.columns)))
