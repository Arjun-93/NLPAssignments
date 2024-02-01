# Task 2.1: Bigram Language Model


# Importing libraries
import numpy as np
from collections import defaultdict

# Class for Bigram Language Model
class BigramLM: 
    def __init__(self):
        self.bigramCounts = defaultdict(lambda: defaultdict(int))  
        # Bigram counts created using defaultdict, 
        # a dictionary that assigns default values to non-existent keys
        self.vocabulary = set() # Vocabulary set created using set() function (a collection of unique elements)
        self.tokenInitiate = '<start>'  # Token for start of sentence
        self.tokenTerminate = '<end>'   # Token for end of sentence

    def tokenize_text(self, text): # Function for tokenizing the text into word list
        return [self.tokenInitiate] + text.split() + [self.tokenTerminate]

    def learn_model(self, corpus): # Function for learning the bigram model from the given corpus 
        for sentence in corpus:
            tokens = self.tokenize_text(sentence)
            for i in range(len(tokens) - 1):
                current_word, next_word = tokens[i], tokens[i + 1]
                self.bigramCounts[current_word][next_word] += 1
                self.vocabulary.add(current_word)

    def calculate_probabilities(self):
        self.bigram_probabilities = defaultdict(dict)
        for current_word, next_word_counts in self.bigramCounts.items():
            total_count = sum(next_word_counts.values())
            for next_word, count in next_word_counts.items():
                probability = count / total_count
                self.bigram_probabilities[current_word][next_word] = probability

    def predict_next_word(self, current_word):
        if current_word not in self.bigram_probabilities:
            return None  # Word not present in training data

        next_word_probs = self.bigram_probabilities[current_word]
        next_words, probabilities = zip(*next_word_probs.items())
        chosen_word = np.random.choice(next_words, p=probabilities)
        return chosen_word

with open('Assignment1\corpus.txt', 'r') as f:
    corpus = f.readlines()

# Creating a bigram model
bigram_model = BigramLM()
bigram_model.learn_model(corpus)
bigram_model.calculate_probabilities()

# Predict the next word given a current word
current_word = "language"
next_word = bigram_model.predict_next_word(current_word)
print(f"The predicted next word after '{current_word}' is '{next_word}'")
