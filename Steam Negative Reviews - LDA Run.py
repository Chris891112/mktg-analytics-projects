#!/usr/bin/env python
# coding: utf-8

# ## Modified Script for The Casting of Frank Stone

# Request: how can we improve overall quality in next 6 month (including all IPs)

# ✔ Summary:
# 
# - **Objective: Understand key negative drivers from players review with little more context**
# - The game's main issues are related to optimization (FPS, bugs), story completeness, and short playtime. 
# The LDA topic model emphasizes unigrams as they convey more clear meaning regarding these key issues.

# ✔ Key Evaluations by Topic:
# 
# - Topic 0: Optimization issues and lack of choice.
# - Topic 1: Weakness in the story and ending.
# - Topic 2: Optimization problems with frame drops, crashes, and bugs.
# - Topic 3: Short playtime and low engagement.

# In[1]:


# List Up items we are going to use, mostly for LDA focusedd

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# In[2]:


# 1. loadd and filter
df = pd.read_csv("translated_frankstone_reviews.csv")
filtered_df = df[df['voted_up'] == False].copy()   


# 

# ## One-line script:  quick re-purpose for DBD - Re-run

# In[10]:


# 2. Expand stopwords
custom_stopwords = set(stopwords.words('english')).union({
    'game', 'character', 'dbd', 'like', 'would', 'could', 'play', 'really',
    'even', 'one', 'time', 'story', 'make', 'get','good'
})

# 3. Define text preprocessing class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()  # Prevent SettingWithCopyWarning
        X['tokens'] = X['review_translated'].apply(self.tokenize)
        return X

    def tokenize(self, text):
        if not isinstance(text, str):
            return []
        lemmatizer = WordNetLemmatizer()
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        tokens = text.lower().split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and word.isalpha()]
        return tokens

# 4. Apply bigrams
bigram = Phrases(filtered_df['review_translated'].dropna().apply(str.split), min_count=5, threshold=10)
bigram_model = Phraser(bigram)

print("🔍 Sample Bigram Phrases:")
for phrase, score in bigram_model.phrasegrams.items():
    print(f"{phrase}: {score}")  # Check if the bigram model is created correctly

# Function to tokenize with bigrams
def tokenize_with_bigrams(text):
    if pd.isna(text):
        return []  # Handle NaN
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    bigram_tokens = list(bigram_model[tokens])  # Apply bigram model
    return bigram_tokens

# Apply bigrams to filtered_df['tokens']
filtered_df['tokens'] = filtered_df['review_translated'].apply(tokenize_with_bigrams)

# Check if bigrams are applied
print("\n🔍 Sample tokenized reviews (with bigrams):")
for tokens in filtered_df['tokens'].head(5):
    print(tokens)  # Confirm bigrams are applied

# 5. Execute LDA
class LDATransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.dictionary = corpora.Dictionary(X['tokens'])
        self.dictionary.filter_extremes(no_below=5, no_above=0.7) #here we just apply arbitrary level control 
        self.corpus = [self.dictionary.doc2bow(tokens) for tokens in X['tokens']]

        # Find the optimal number of topics (2 to 10)
        best_score = -1
        best_topics = 2
        best_model = None

        for num_topics in range(2, 11):
            lda_model = LdaModel(self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=30, iterations=400, alpha='auto', eta='auto')
            coherence_model_lda = CoherenceModel(model=lda_model, texts=X['tokens'], dictionary=self.dictionary, coherence='c_v')
            coherence_score = coherence_model_lda.get_coherence()

            if coherence_score > best_score:
                best_score = coherence_score
                best_topics = num_topics
                best_model = lda_model

        self.lda_model = best_model
        self.best_topics = best_topics
        print(f"\nOptimal number of topics: {self.best_topics}, Coherence Score: {best_score}")

        return self

    def transform(self, X):
        corpus = [self.dictionary.doc2bow(tokens) for tokens in X['tokens']]
        topic_assignments = [
            max(self.lda_model.get_document_topics(doc), key=lambda x: x[1])[0] if doc else None for doc in corpus
        ]
        X = X.copy()  # Prevent SettingWithCopyWarning
        X['assigned_topic'] = topic_assignments
        return X

# Create the pipeline for quick readability
pipeline = Pipeline(steps=[
    ('preprocessor', TextPreprocessor()),
    ('lda_transformer', LDATransformer())
])

processed_df = pipeline.fit_transform(filtered_df)


# + new added --> LDA topic and keyword weight 
topics = pipeline.named_steps['lda_transformer'].lda_model.show_topics(num_words=10, formatted=False)

print("\n📌 LDA Topic Word Probabilities:")
for topic_id, topic_words in topics:
    print(f"\n🔹 Topic {topic_id}:")
    for word, weight in topic_words:
        print(f"   {word}: {weight:.4f}")


# 6. Visualize topic distribution by country
language_topic_counts = processed_df.groupby('language')['assigned_topic'].value_counts().unstack()

plt.figure(figsize=(12, 6))
ax = language_topic_counts.plot(kind='bar', stacked=True, cmap='coolwarm', alpha=0.8)

plt.title("Topic Distribution by Country", fontsize=16)
plt.xlabel("Language", fontsize=14)
plt.ylabel("Review Count", fontsize=14)
plt.xticks(rotation=45)
plt.legend(title="Topic ID")
plt.tight_layout()

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='center', fontsize=6, color='black')

plt.show()

# 7. Generate WordCloud reflecting bigrams
def bigram_wordcloud(lda_model, num_topics=4):
    plt.figure(figsize=(12, 6))

    for i in range(num_topics):
        plt.subplot(2, 2, i + 1)
        topic_terms = dict(lda_model.show_topic(i, 15))

        # Combine bigram words with underscores
        bigram_terms = {word.replace(" ", "_"): weight for word, weight in topic_terms.items()}

        wordcloud = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(bigram_terms)

        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {i}")

    plt.tight_layout()
    plt.show()

bigram_wordcloud(pipeline.named_steps['lda_transformer'].lda_model, num_topics=pipeline.named_steps['lda_transformer'].best_topics)

# 8. Visualize number of reviews per topic
topic_counts = processed_df['assigned_topic'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
ax = topic_counts.plot(kind='bar', color='lightcoral', alpha=0.8)

plt.title("Number of Reviews per Topic", fontsize=16)
plt.xlabel("Topic ID", fontsize=14)
plt.ylabel("Review Count", fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=12, color='black')

plt.show()


# In[11]:


filtered_df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




