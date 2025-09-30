def preprocessing(text):
    text = text.lower()
    tokens = text.split()
    return tokens


def unigram_and_bigram(corpus_path):
    unigram_counts = {}
    bigram_counts = {}
    with open(corpus_path, "r") as f:
        for review in f:
            tokens = preprocessing(review.strip())
            # unigrams
            for token in tokens:
                unigram_counts[token] = unigram_counts.get(token, 0) + 1
            # bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
    
    return unigram_counts, bigram_counts


def computing_probabilities(unigram_counts, bigram_counts):
    total_unigrams = sum(unigram_counts.values())
    
    # unigram
    unigram_prob = {}
    for word, count in unigram_counts.items():
        unigram_prob[word] = count / total_unigrams
    
    # bigram
    bigram_prob = {}
    for (word1, word2), count in bigram_counts.items():
        if word1 not in bigram_prob:
            bigram_prob[word1] = {}
        bigram_prob[word1][word2] = count / unigram_counts[word1]
    
    return unigram_prob, bigram_prob


if __name__ == "__main__":
    train_file = "A1_DATASET/train.txt"
    unigram_counts, bigram_counts = unigram_and_bigram(train_file)
    unigram_prob, bigram_prob = computing_probabilities(unigram_counts, bigram_counts)

    # printing top 10 unigrams
    print("Top 10 unigram probabilities:")
    for word, prob in list(unigram_prob.items())[:10]:
        print(f"P({word}) = {prob:.4f}")

    # printing bigram probabilities
    test_word = "the"
    if test_word in bigram_prob:
        print(f"\nBigram probabilities for context '{test_word}':")
        for word, prob in list(bigram_prob[test_word].items())[:10]:
            print(f"P({word}|{test_word}) = {prob:.4f}")
