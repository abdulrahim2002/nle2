# this file is used for importing `generate_text_ngrams_model` in `task2`
# most of the code is just copied over from `task1.ipynb``
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('imdb_reviews.csv')
unigram_vectorizer = CountVectorizer(ngram_range=(1,1))
bigram_vectorizer =  CountVectorizer(ngram_range=(2,2))
trigram_vectorizer = CountVectorizer(ngram_range=(3,3))
fourgram_vectorizer = CountVectorizer(ngram_range=(4,4))

X_unigram = unigram_vectorizer.fit_transform(df['review'])
X_bigram  = bigram_vectorizer.fit_transform(df['review'])
X_trigram = trigram_vectorizer.fit_transform(df['review'])
X_fourgram = fourgram_vectorizer.fit_transform(df['review'])


# maps the value of n -> frequency map.
ngram_freq_maps = {}

ngram_freq_maps[1] = dict(zip(
    unigram_vectorizer.get_feature_names_out(),
    X_unigram.sum(axis=0).A1
))

ngram_freq_maps[2] = dict(zip(
    bigram_vectorizer.get_feature_names_out(),
    X_bigram.sum(axis=0).A1
))

ngram_freq_maps[3] = dict(zip(
    trigram_vectorizer.get_feature_names_out(),
    X_trigram.sum(axis=0).A1
))

ngram_freq_maps[4] = dict(zip(
    fourgram_vectorizer.get_feature_names_out(),
    X_fourgram.sum(axis=0).A1
))

# compute conditional prob dictionaries from freq maps
ngram_prob_maps = {}

# for 2,3,4-grams
for n in range(2, 5):
    prob_dict = {}

    for ngram, count in ngram_freq_maps[n].items():
        # suppose the tokens are `cat is cute`
        tokens = ngram.split()

        # context is `cat is` and next_word is `cute`
        context = " ".join(tokens[:-1])
        next_word = tokens[-1]

        # Count(`cat is`)
        context_count = ngram_freq_maps[n-1].get(context, 0)

        if context_count > 0:
            # P(`cute` | `cat is`) = Count(`cat is cute`) / Count(`cat is`)
            prob = count / context_count

            if context not in prob_dict:
                prob_dict[context] = {}

            # ngram_prob_maps[3][`cat is`] = { `cute`: number_found_above }
            prob_dict[context][next_word] = prob

    ngram_prob_maps[n] = prob_dict

# For unigrams (fallback), normalize to probs
total_unigrams = sum(ngram_freq_maps[1].values())
unigram_probs = {word: freq / total_unigrams for word, freq in ngram_freq_maps[1].items()}


def generate_text_ngrams_model(n: int, initial_prefix: str, max_length: int=50):
    """
        Generate text using n-grams model. Assume that initial n-1 words are
        given as prefix. Uses backoff strategy to shorter n-grams if larger
        n-grams are not found.
    """
    generated = initial_prefix.split()
    current_prefix = initial_prefix

    for _ in range(max_length - len(generated)):
        next_word = None

        prefix_tokens = current_prefix.split()

        for backoff_level in range(len(prefix_tokens) + 1):  # 0: full, 1: remove first

            # suppose we have current prefix "the movie"
            test_prefix = " ".join(prefix_tokens[backoff_level:])

            # check if "the movie" is in ngrams_prob_maps[3]
            # if no -> increase backoff, make the prefix "movie" next time

            if not test_prefix:
                # the test prefix is empty, maybe because we do not have enough tokens left after backoff
                # in this case we randomly select from unigram probabilities
                next_word = random.choices(list(unigram_probs.keys()), weights=list(unigram_probs.values()))[0]
                break

            elif test_prefix in ngram_prob_maps[n]:
                # "the movie" is found in ngram_prob_maps[3]
                probs = ngram_prob_maps[n][test_prefix]
                # we will have probs like: {  "was": 0.6, "is": 0.4 }
                next_word = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
                # we choose one of these words
                break

        if not next_word:
            break  # No options, halt generation

        generated.append(next_word)

        # Update prefix to last N-1 words
        current_prefix = " ".join(generated[-(n-1):])

    return " ".join(generated)
