## NLE assignment 2

Solution for `studienleistung_2`.

### Task 1

file: `task1.ipynb`

- In task 1, we implement text generation using `n-grams model`.
- we implement `backoff` i.e. when probabilities for higher `n-grams` are not available
    we try lower order `n-grams`
- lastly, we generate 5000 samples of artificial reviews using our n-grams language model

### Task 2

file: `task2.pynb`

- In task 2, we implement a Naive Bayes classifier using 3-grams as
    features to distinguish between original ("orig") and generated ("gen")
    reviews.
- Initially we train the classifier on 5,000 original reviews from the corpus and
    5,000 artificially generated reviews using the N-grams model from Task 1.
- For evaluation, we run a 5,000-iteration online learning loop: randomly sample
    an unseen original review or generate a new review, classify it, and
    if misclassified, add it to the training data to update the model.
- We report the classifier's accuracy on a held-out test set every 100 iterations
    and analyze the performance trend over time.


### EDA

file: `eda.ipynb`

- For each N-gram type, we compute frequency mappings, sort by frequency, and plot
    histograms of the frequency distributions on a log scale to visualize word/N-gram usage patterns.
