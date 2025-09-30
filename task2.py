def preprocessing(corpus_path, unk_threshold=1):
    unigram_counts = {}
    tokenized_sentences = []
    with open(corpus_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            tokens = [t.lower() for t in tokens]
            
            for token in tokens:
                unigram_counts[token] = unigram_counts.get(token, 0) + 1
            
            tokenized_sentences.append(tokens)

    # vocabulary with <UNK>
    vocab = set()
    for word, count in unigram_counts.items():
        if count > unk_threshold:
            vocab.add(word)
    vocab.add("<UNK>")

    # rare words to <UNK>
    modified_sentences = []
    for sentence in tokenized_sentences:
        processed = []
        for word in sentence:
            if word in vocab:
                processed.append(word)
            else:
                processed.append("<UNK>")
        modified_sentences.append(processed)

    return modified_sentences, vocab



def count_ngrams(sentences):
    unigram_counts = {}
    bigram_counts = {}
    for tokens in sentences:
        tokens = ["<s>"] + tokens + ["</s>"]
        
        # unigrams
        for token in tokens:
            unigram_counts[token] = unigram_counts.get(token, 0) + 1
        
        # bigrams
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    return unigram_counts, bigram_counts


def bigram_laplace(bigram_counts, unigram_counts, vocab_size):
    bigram_probs = {}
    for (word1, word2), count in bigram_counts.items():
        if word1 not in bigram_probs:
            bigram_probs[word1] = {}
        bigram_probs[word1][word2] = (count + 1) / (unigram_counts[word1] + vocab_size)
    return bigram_probs


def bigram_addk(bigram_counts, unigram_counts, vocab_size, k=0.5):
    bigram_probs = {}
    for word1 in unigram_counts:
        if word1 not in bigram_probs:
            bigram_probs[word1] = {}
        for word2 in unigram_counts:
            count = bigram_counts.get((word1, word2), 0)
            bigram_probs[word1][word2] = (count + k) / (unigram_counts[word1] + k * vocab_size)
    return bigram_probs


if __name__ == "__main__":
    train_file = "A1_DATASET/train.txt"
    sentences, vocab = preprocessing(train_file, unk_threshold=1)
    vocab_size = len(vocab)

    # n-grams
    unigram_counts, bigram_counts = count_ngrams(sentences)

    # bigram
    bigram_laplace = bigram_laplace(bigram_counts, unigram_counts, vocab_size)
    bigram_addk = bigram_addk(bigram_counts, unigram_counts, vocab_size, k=0.5)

    # printing top 10 unigram
    total_unigrams = sum(unigram_counts.values())
    unigram_probs = {}
    for word, count in unigram_counts.items():
        unigram_probs[word] = count / total_unigrams

    print("Top 10 unigram probabilities:")
    for word, prob in list(unigram_probs.items())[:10]:
        print(f"P({word}) = {prob:.4f}")

    # printing bigram for word 'the'
    word = "the"
    if word in bigram_laplace:
        print(f"\nBigram probabilities for context '{word}' (Laplace):")
        for word1, prob in list(bigram_laplace[word].items())[:10]:
            print(f"P({word1}|{word}) = {prob:.4f}")

    if word in bigram_addk:
        print(f"\nBigram probabilities for context '{word}' (Add-0.5):")
        for word1, prob in list(bigram_addk[word].items())[:10]:
            print(f"P({word1}|{word}) = {prob:.4f}")
