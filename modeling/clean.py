import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.getcwd())

# importing raw data
df_med = pd.read_csv('/Users/faes/ds-bootcamp/version1 ds-capstone-project-template/notebooks/medquad.csv')


df_med.head(7)

df_med.info

df_med.describe()

df_med.shape

# check how many duplicated rows exist in the data frame
df_med.duplicated().value_counts()

# check data types in data frame
df_med.dtypes

# display number of distinct elements
df_med.focus_area.nunique()

# ## Missing Data

# import missingno
import missingno as msno

# display number of missing values per column
df_med.isna().sum()

df = df_med.copy()

df_cleaned = df.dropna(subset=['answer'])

df_cleaned = df_cleaned.dropna(subset=['focus_area']) 

df = df_cleaned

df['focus_area'].value_counts(dropna=False)

df['source'].value_counts()

df['source'].value_counts().plot(kind='bar')

df['question_len'] = df['question'].fillna('').apply(lambda x: len(x.split()))
df['answer_len'] = df['answer'].fillna('').apply(lambda x: len(x.split()))

print(df[['question_len', 'answer_len']].describe())

import matplotlib.pyplot as plt

plt.hist(df['question_len'], bins=50, alpha=0.6, label='Question length')
plt.hist(df['answer_len'], bins=50, alpha=0.6, label='Answer length')
plt.legend()
plt.show()

top_focus = df['focus_area'].value_counts().nlargest(10)
top_focus.plot(kind='bar')

print(df[df['answer'].isnull()])


# Questions or answers with very few words (e.g., less than 3)
short_questions = df[df['question'].fillna('').apply(lambda x: len(x.split()) < 3)]
short_answers = df[df['answer'].fillna('').apply(lambda x: len(x.split()) < 3)]

print("Short questions:\n", short_questions[['question', 'answer']])
print("Short answers:\n", short_answers[['question', 'answer']])

df = df[df['answer'].fillna('').apply(lambda x: len(x.split()) >= 3)]
df = df[df['question'].fillna('').apply(lambda x: len(x.split()) >= 3)]


missing_questions = df[df['question'].isnull()]
missing_answers = df[df['answer'].isnull()]

print(f"Missing questions: {len(missing_questions)}")
print(f"Missing answers: {len(missing_answers)}")


# duplicates = df[df.duplicated(subset=['question', 'answer'], keep=False)]
# print(f"Number of duplicate question-answer pairs: {len(duplicates)}")

# print(duplicates.index.tolist())


import re

# Answers with less than 50% alphabetic characters (maybe gibberish or code)
def alpha_ratio(text):
    text = str(text)
    if len(text) == 0:
        return 0
    alphabets = len(re.findall(r'[a-zA-Z]', text))
    return alphabets / len(text)

noisy_answers = df[df['answer'].apply(alpha_ratio) < 0.5]
print(f"Potentially noisy answers: {len(noisy_answers)}")
print(noisy_answers[['question', 'answer']].head())


df = df[df['answer'].apply(alpha_ratio) >= 0.5]


import matplotlib.pyplot as plt

df['answer_len'] = df['answer'].fillna('').apply(lambda x: len(x.split()))

plt.boxplot(df['answer_len'])
plt.title('Answer Length Distribution')
plt.show()


# # Extra
# # 6. Find and display the 20 most common words in the 'question' column (after removing stopwords and punctuation)

# from collections import Counter
# import nltk
# from nltk.corpus import stopwords
# import string

# # Download stopwords if not already downloaded
# nltk.download('stopwords')

# stop_words = set(stopwords.words('english'))

# # Combine all questions into one large string, convert to lowercase
# all_questions = " ".join(df_med['question'].fillna('').str.lower())

# # Remove punctuation
# all_questions = all_questions.translate(str.maketrans('', '', string.punctuation))

# # Split into words
# words = all_questions.split()

# # Remove stopwords
# filtered_words = [word for word in words if word not in stop_words]

# # Count the most common words
# word_counts = Counter(filtered_words)
# most_common_words = word_counts.most_common(20)

# print("Top 20 most common words in questions:")
# for word, count in most_common_words:
#     print(f"{word}: {count}")

