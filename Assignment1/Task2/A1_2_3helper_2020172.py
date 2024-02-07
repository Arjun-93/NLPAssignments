from data.utils import emotion_scores
import pandas as pd
from A1_2_1_2_2020172 import BigramLM

corpus = pd.read_csv('corpus.txt')
emotion = pd.read_csv('total_emotion_scores.txt')
# data = pd.concat([corpus, emotion], axis=1)

# Creating a bigram model
bigram_model = BigramLM()
bigram_model.learn_model(corpus['text'])
bigram_model.calculate_probabilities()


print(len(bigram_model.bigramCounts))
with open('bigram_counts.txt', 'w') as f:
    f.write("word1 word2 emotion \n")

for current_word, next_word_counts in bigram_model.bigramCounts.items():
    for next_word, count in next_word_counts.items():
        with open('bigram_counts.txt', 'a') as f:
            f.write(f"{current_word}-{next_word}-{emotion_scores(next_word)} \n")
        # print(f"{current_word} {next_word} {count}")
        # print(emotion_scores(next_word))
        # print("\n")