from collections import Counter
import re
import os

class BigramLM:
    def __init__(self):
        self.bigramCounts = Counter()
        self.unigramCounts = Counter()

    def train(self, corpus):
        for sentence in corpus:
            words = re.findall(r'\b\w+\b', sentence.lower())  # Tokenize and convert to lowercase
            bigrams = zip(words, words[1:])
            self.bigramCounts.update(bigrams)
            self.unigramCounts.update(words)

    def laplace_smoothing_probability(self, bigram):
        numerator = self.bigramCounts[bigram] + 1
        denominator = self.unigramCounts[bigram[0]] + len(set(self.unigramCounts))  # Laplace smoothing
        probability = numerator / denominator
        return probability

    def kneser_ney_smoothing_probability(self, bigram):
        discount = 0.75  # You can adjust this parameter
        continuation_count = Counter()

        for other_bigram in self.bigramCounts:
            if other_bigram[0] == bigram[0]:
                continuation_count.update([other_bigram[1]])

        numerator = max(self.bigramCounts[bigram] - discount, 0) + discount * len(continuation_count) / len(self.bigramCounts)
        denominator = self.unigramCounts[bigram[0]]
        probability = numerator / denominator

        return probability

    def get_top_n_bigrams_with_probabilities(self, n=None, smoothing=None):
        top_n_bigrams = []
        for bigram, _ in self.bigramCounts.most_common(n):
            probability = 0
            if smoothing == 'kneser_ney':
                probability = self.kneser_ney_smoothing_probability(bigram)
            else:
                probability = self.laplace_smoothing_probability(bigram)
            top_n_bigrams.append((bigram, probability))
        return top_n_bigrams
    
def get_top_n_bigram_with_probabilities(corpus, n=None):
    bigram_counts = Counter()
    unigram_counts = Counter()

    for sentence in corpus:
        words = re.findall(r'\b\w+\b', sentence.lower())  # Tokenize and convert to lowercase
        bigrams = zip(words, words[1:])
        bigram_counts.update(bigrams)
        unigram_counts.update(words)

    top_n_bigrams = []
    total_bigrams = sum(bigram_counts.values())

    for bigram, count in bigram_counts.most_common(n):
        probability = count / total_bigrams
        top_n_bigrams.append((bigram, probability))

    return top_n_bigrams

getCwd = os.getcwd()

with open(getCwd + '\\Assignment1\\data\\corpus.txt', 'r') as file:
    corpus = file.readlines()
bigram_model = BigramLM()
bigram_model.train(corpus)

# Without smoothing
top_bigrams = get_top_n_bigram_with_probabilities(corpus, n=5)
print("-------------------------------------------------------------------------------------------------------------------------------------------")
print("Top Bigrams without Smoothing:", top_bigrams)
print()

# Laplace smoothing
top_bigrams_laplace = bigram_model.get_top_n_bigrams_with_probabilities(n=5, smoothing='laplace')
print("Top Bigrams with Laplace Smoothing:", top_bigrams_laplace)
print()

# Kneser-Ney smoothing (placeholder, needs implementation)
top_bigrams_kneser_ney = bigram_model.get_top_n_bigrams_with_probabilities(n=5, smoothing='kneser_ney')
print("Top Bigrams with Kneser-Ney Smoothing:", top_bigrams_kneser_ney)
print("-------------------------------------------------------------------------------------------------------------------------------------------")