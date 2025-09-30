# TASK 3:
import math

def preprocess_with_unk(corpus_path, unk_threshold=1):
    unigram_counts = {}
    tokenized_sentences = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().lower().split()
            tokenized_sentences.append(tokens)
            for token in tokens:
                unigram_counts[token] = unigram_counts.get(token, 0) + 1

    vocab = {word for word, count in unigram_counts.items() if count > unk_threshold}
    vocab.add("<UNK>")

    processed_sentences = []
    for sentence in tokenized_sentences:
        processed = [w if w in vocab else "<UNK>" for w in sentence]
        processed_sentences.append(processed)

    return processed_sentences, vocab

def count_ngrams(sentences):

    unigram_counts = {}
    bigram_counts = {}

    for tokens in sentences:
        tokens = ["<s>"] + tokens + ["</s>"]

        for token in tokens:
            unigram_counts[token] = unigram_counts.get(token, 0) + 1

        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i + 1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    return unigram_counts, bigram_counts

def calculate_perplexity(validation_path, model_bigram_counts, model_unigram_counts, vocab, k=1.0):
    vocab_size = len(vocab)
    log_prob_sum = 0.0
    N = 0

    with open(validation_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_tokens = line.strip().lower().split()
            processed = [word if word in vocab else "<UNK>" for word in raw_tokens]
            tokens = ["<s>"] + processed + ["</s>"]

            N += len(tokens) - 1

            for i in range(1, len(tokens)):
                w1 = tokens[i-1]
                w2 = tokens[i]


                bigram_count = model_bigram_counts.get((w1, w2), 0)
                unigram_count = model_unigram_counts.get(w1, 0)

                denominator = unigram_count + k * vocab_size
                if denominator == 0:
                    prob = 1.0 / vocab_size
                else:
                    prob = (bigram_count + k) / denominator

                log_prob_sum += math.log(prob, 2)

    if N == 0:
        return float('inf')

    avg_log_prob = log_prob_sum / N
    perplexity = math.pow(2, -avg_log_prob)

    return perplexity


if __name__ == "__main__":

    train_file = "/content/train.txt"
    validation_file = "/content/val.txt"

    print(f"Training model on: {train_file}")

    train_sentences, vocab = preprocess_with_unk(train_file, unk_threshold=1)

    model_unigram_counts, model_bigram_counts = count_ngrams(train_sentences)
    print(f"Vocabulary size: {len(vocab)}")

    print(f"\nCalculating perplexity for: {validation_file}")

    perplexity_laplace = calculate_perplexity(
        validation_file,
        model_bigram_counts,
        model_unigram_counts,
        vocab,
        k=1.0
    )
    print(f"Perplexity with Laplace (k=1.0) smoothing: {perplexity_laplace:.4f}")

    perplexity_add_k = calculate_perplexity(
        validation_file,
        model_bigram_counts,
        model_unigram_counts,
        vocab,
        k=0.5
    )
    print(f"Perplexity with Add-k (k=0.5) smoothing: {perplexity_add_k:.4f}")