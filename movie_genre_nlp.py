# ===== 1. Imports =====
import math
import pandas as pd
import nltk
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ===== 2. Download NLTK resources =====
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# ===== 3. Load data =====
train = pd.read_csv("kaggle_movie_train.csv")
test = pd.read_csv("kaggle_movie_test.csv")

# ===== 4. Segmentation (text -> sentences) =====
train["sentences"] = train["text"].apply(sent_tokenize)
test["sentences"] = test["text"].apply(sent_tokenize)
print(train["sentences"].str.len().describe())

# ===== 5. Tokenization (lowercase + split into tokens) =====
def tokenize(sentences):
    tokens = []
    for sentence in sentences:
        tokens.extend(word_tokenize(sentence.lower()))
    return tokens

train["tokens"] = train["sentences"].apply(tokenize)
test["tokens"] = test["sentences"].apply(tokenize)
print(train["tokens"].str.len().mean())

# ===== 6. Stop words removal =====
stop_words = set(stopwords.words("english"))

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]

train["no_stop"] = train["tokens"].apply(remove_stopwords)
test["no_stop"] = test["tokens"].apply(remove_stopwords)
print(train["no_stop"].str.len().mean())

# ===== 7. Remove punctuation and numbers =====
def keep_alpha(tokens):
    return [t for t in tokens if t.isalpha()]

train["clean"] = train["no_stop"].apply(keep_alpha)
test["clean"] = test["no_stop"].apply(keep_alpha)
print(train["clean"].str.len().mean())

# ===== 8. Stemming =====
stemmer = PorterStemmer()

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

train["stemmed"] = train["clean"].apply(stem_tokens)
test["stemmed"] = test["clean"].apply(stem_tokens)

# ===== 9. Final cleaned text =====
train["final_text"] = train["stemmed"].apply(" ".join)
test["final_text"] = test["stemmed"].apply(" ".join)
print(train[["genre", "final_text"]].head())

# ===== 10. Add start/end markers to each document =====
train["seq"] = train["stemmed"].apply(lambda t: ["<s>"] + t + ["</s>"])
test["seq"] = test["stemmed"].apply(lambda t: ["<s>"] + t + ["</s>"])

# ===== 11. Count unigrams, bigrams, trigrams (train only) =====
unigrams, bigrams, trigrams = Counter(), Counter(), Counter()

for seq in train["seq"]:
    unigrams.update(seq)
    bigrams.update(zip(seq, seq[1:]))
    trigrams.update(zip(seq, seq[1:], seq[2:]))

V = len(unigrams)
N = sum(unigrams.values())

print("Unigrams (V):", V)
print("Bigrams:", len(bigrams))
print("Trigrams:", len(trigrams))
print("Total tokens (N):", N)

# ===== 12. Most frequent n-grams =====
print(pd.DataFrame(unigrams.most_common(10), columns=["unigram", "count"]))
print(pd.DataFrame(bigrams.most_common(10), columns=["bigram", "count"]))
print(pd.DataFrame(trigrams.most_common(10), columns=["trigram", "count"]))

# ===== 13. Probability functions (MLE and Laplace) =====
def p_uni(w):
    return unigrams[w] / N

def p_bi(prev, w):
    return bigrams[(prev, w)] / unigrams[prev]

def p_uni_laplace(w):
    return (unigrams[w] + 1) / (N + V)

def p_bi_laplace(prev, w):
    return (bigrams[(prev, w)] + 1) / (unigrams[prev] + V)

# ===== 14. Step-by-step calculation on first 7 tokens of document 0 =====
sample = train["seq"][0][:7]
rows = []

for prev, w in zip(sample, sample[1:]):
    p = p_bi(prev, w)
    rows.append((prev, w, bigrams[(prev, w)], unigrams[prev], p, math.log10(p)))

steps = pd.DataFrame(rows, columns=["prev", "word", "C(prev,word)", "C(prev)", "P(word|prev)", "log10 P"])
print(sample)
print(steps.to_string())
print("Sum of log10 P =", steps["log10 P"].sum())
print("Product of P   =", 10 ** steps["log10 P"].sum())

# ===== 15. Probability of a full document (log10) =====
def doc_logp_bi(seq, p):
    return sum(math.log10(p(a, b)) for a, b in zip(seq, seq[1:]))

def doc_logp_uni(seq, p):
    return sum(math.log10(p(b)) for b in seq[1:])

doc = train["seq"][0]
lp = doc_logp_bi(doc, p_bi)
print("Tokens predicted:", len(doc) - 1)
print("log10 P(doc 0) =", lp)
print("P(doc 0) = 10 ^", lp)

# ===== 16. Probability and perplexity of the whole train set =====
def evaluate(data, logp):
    total, n = 0, 0
    for seq in data["seq"]:
        total += logp(seq)
        n += len(seq) - 1
    return total, n, 10 ** (-total / n)

results = []
for name, data, logp in [
    ("Train - Unigram", train, lambda s: doc_logp_uni(s, p_uni)),
    ("Train - Bigram (Markov)", train, lambda s: doc_logp_bi(s, p_bi)),
]:
    total, n, ppl = evaluate(data, logp)
    results.append((name, total, n, ppl))

print(pd.DataFrame(results, columns=["model", "log10 P(dataset)", "tokens", "perplexity"]).to_string())

# ===== 17. Probability and perplexity of the test set (Laplace smoothing) =====
results = []
for name, data, logp in [
    ("Test - Unigram + Laplace", test, lambda s: doc_logp_uni(s, p_uni_laplace)),
    ("Test - Bigram + Laplace", test, lambda s: doc_logp_bi(s, p_bi_laplace)),
]:
    total, n, ppl = evaluate(data, logp)
    results.append((name, total, n, ppl))

print(pd.DataFrame(results, columns=["model", "log10 P(dataset)", "tokens", "perplexity"]).to_string())
