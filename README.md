# Movie Genre NLP: Preprocessing & N-gram Language Model

Text preprocessing and a Markov (n-gram) language model on the Kaggle
[Movie Genre](https://www.kaggle.com/datasets/lykin22/movie-genre) dataset
(movie script excerpts labeled by genre).

## What it does

1. **Text preprocessing**: sentence segmentation, tokenization (lowercased), stop words removal, punctuation and number removal, Porter stemming.
2. **N-grams**: counts unigrams, bigrams and trigrams on the training set, with `<s>` / `</s>` markers per document.
3. **Markov assumption**: computes `P(w_i | w_{i-1}) = C(w_{i-1}, w_i) / C(w_{i-1})`, shows the step-by-step calculation on a sample document, and computes the log probability and perplexity of the dataset.
4. **Laplace smoothing**: used on the test set to avoid zero probabilities.

## Project structure

```
NLP_Project/
├── movie_genre_nlp.py
├── kaggle_movie_train.csv
├── kaggle_movie_test.csv
└── README.md
```

## Setup

Requires Python 3.8+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas nltk
```

NLTK data (`punkt`, `punkt_tab`, `stopwords`) is downloaded automatically on the first run.

## Run

```bash
python movie_genre_nlp.py
```

Takes about 2 to 5 minutes and needs roughly 4 GB of RAM.

## Results

| Model | Perplexity |
|---|---|
| Train, Unigram | 3,552 |
| Train, Bigram (Markov) | 216.8 |
| Test, Unigram + Laplace | 3,627 |
| Test, Bigram + Laplace | 14,707 |

The bigram model fits the training set far better, but on the test set it is
worse than the unigram model. Most test bigrams are unseen in training, and
add-one smoothing over a vocabulary of about 41k words spreads too much
probability onto them. Better smoothing (add-k, interpolation, backoff) is the
natural next step.

## Dataset

Only `kaggle_movie_train.csv` (`id`, `text`, `genre`) and
`kaggle_movie_test.csv` (`id`, `text`) are used. The metadata and plot summary
files in the archive are not linked to these ids and are ignored.
