import pandas as pd
import numpy as np
import random
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



intents = {
    "greeting": ["hello","hi","hey","good morning","salam"],
    "goodbye": ["bye","goodbye","see you","allah hafiz"],
    "admission": ["are admissions open","how to apply","admission process"],
    "programs": ["what programs do you offer","courses available"],
    "cs_programs": ["bs cs","ai program","software engineering"],
    "fees": ["fee structure","tuition fee","semester fee"],
    "location": ["where is campus","address"],
    "contact": ["contact number","helpline"]
}

data = []

for intent, texts in intents.items():
    for _ in range(60):
        data.append([random.choice(texts), intent])

# data of university
extra = [
    ["bs artificial intelligence", "cs_programs"],
    ["bs software engineering", "cs_programs"],
    ["biochemistry program", "health_programs"],
    ["bba hons", "business_programs"],
    ["bs english", "arts_programs"],
    ["livestock diploma", "veterinary_programs"],
    ["sahiwal campus address", "location"],
    ["helpline 0335 111 8383", "contact"]
]

data.extend(extra)

df = pd.DataFrame(data, columns=["text", "intent"])


# TRAIN MODEL


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["intent"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)


# SAVE MODEL


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")