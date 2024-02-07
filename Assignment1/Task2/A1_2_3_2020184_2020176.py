# Importing libraries
import numpy as np
import pandas as pd
import random
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

class BigramLM: 
    '''
    BigramLM: Provides the functional definition of a bigram language model.
    '''
    def __init__(self):
        '''
        BRIEF:      Constructor for the creation of bigram LM.
        PARAMETERS: None.
        '''
        self.bigramCounts = defaultdict(lambda: defaultdict(int))  
        self.vocabulary = defaultdict(int) # Vocabulary for learning
        self.tokenInitiate = '<s>'  # Token for start of sentence
        self.tokenTerminate = '</s>'   # Token for end of sentence
        self.bigram_probabilities = defaultdict(lambda: defaultdict(int))

    def __tokenize_text(self, text):
        '''
        BRIEF:      (Internal method) Tokenizing the sentence into a word list.
        PARAMETERS: The sentence to be tokenized (text).
        RETURN:     list of tokens in the sentence.
        '''
        return [self.tokenInitiate] + text.split() + [self.tokenTerminate]

    def emotion_score_probability(self, current_word, next_word, emotion_scores, emotion):
        # self.emotion_score_probability = defaultdict(dict)
        # # self.emotion_score_probability[word] = float(emotion_score)
        # for current_word, next_word_counts in self.bigramCounts.items():
        #     for next_word in next_word_counts:
        #         scores = str(emotion_scores[emotion_scores['word1'] == current_word][emotion_scores['word2'] == next_word]['emotion'].values)
        #         # Remove brackets and split the string into individual items
        #         items_str = scores.strip('[]')[2:-3].split('}, ')
        #         items_str = [item + '}' for item in items_str]
        #         items_str[-1] = items_str[-1][:-1]
        #         # Convert each item string to a dictionary
        #         emotions = [eval(item) for item in items_str]
        #         # Find the score associated with the label
        #         emotion_score = 0
        #         for item in emotions:
        #             if item['label'] == emotion.lower():
        #                 emotion_score = item['score']
        #         # Print the score
        #         # print(f"Score for {emotion}:", emotion_score)
        #         self.bigram_probabilities[current_word][next_word] = (self.bigram_probabilities[current_word][next_word] + emotion_score) / 2
                
        # # self.calculate_bigram_probabilities()
        #### CALCULATE EMOTION SCORE CORRESPONDING TO "emotion" FOR THE BIGRAM ####
        scores = str(emotion_scores[emotion_scores['word1'] == current_word][emotion_scores['word2'] == next_word]['emotion'].values)
        # Remove brackets and split the string into individual items
        items_str = scores.strip('[]')[2:-3].split('}, ')
        items_str = [item + '}' for item in items_str]
        items_str[-1] = items_str[-1][:-1]
        if items_str == ['']:
            items_str = ["{'label': 'sadness', 'score': 0}", "{'label': 'joy', 'score': 0}", "{'label': 'love', 'score': 0}", "{'label': 'anger', 'score': 0}", "{'label': 'fear', 'score': 0}", "{'label': 'surprise', 'score': 0}"]
        # Convert each item string to a dictionary
        # print(f'{current_word}  {next_word}')
        emotions = [eval(item) for item in items_str]
        # Find the score associated with the label
        emotion_score = 0
        for item in emotions:
            if item['label'] == emotion.lower():
                emotion_score = item['score']
        return emotion_score
        ###########################################################################

    def __simple_probabilities(self, current_word, next_word_candidates, emotion=None, emotion_scores=None):
        '''
        BRIEF:      (Internal method) Calculating simple bigram probabilities.
        PARAMETERS: Last word (current_word) | Candidates for the next word (next_word_candidates).
        '''
        for next_word in next_word_candidates:
            if next_word != '<s>':
                self.bigram_probabilities[current_word][next_word] = (self.bigramCounts[current_word][next_word]) / (sum(self.bigramCounts[current_word].values()))
                if emotion != None:
                    emotion_score = self.emotion_score_probability(current_word, next_word, emotion_scores, emotion)
                    self.bigram_probabilities[current_word][next_word] = (self.bigram_probabilities[current_word][next_word] + emotion_score) / 2
            else:
                continue

    def __laplace_probabilities(self, current_word, next_word_candidates, emotion=None, emotion_scores=None):
        '''
        BRIEF:      (Internal method) Computing the Laplace probabilities given the last word.
        PARAMETERS: Last word (current_word) | Candidates for the next word (next_word_candidates).
        '''
        for next_word in next_word_candidates:
            self.bigram_probabilities[current_word][next_word] = (self.bigramCounts[current_word][next_word] + 1) / (sum(self.bigramCounts[current_word].values()) + len(next_word_candidates))
            if emotion != None:
                emotion_score = self.emotion_score_probability(current_word, next_word, emotion_scores, emotion)
                self.bigram_probabilities[current_word][next_word] = (self.bigram_probabilities[current_word][next_word] + emotion_score) / 2
          
    def __knesser_ney_smoothing(self):
        '''
        BRIEF:      (Internal method) Performing Knesser-Ney smoothing.
        PARAMETERS: None.
        '''
        pass

    def fit(self, corpus):
        '''
        BRIEF:      Learning the bigram model from the given corpus.
        PARAMETERS: The corpus of samples for learning (corpus).
        ''' 
        for sentence in corpus:
            tokens = self.__tokenize_text(sentence)
            for i in range(len(tokens) - 1):
                current_word, next_word = tokens[i], tokens[i + 1]
                self.bigramCounts[current_word][next_word] += 1
                self.vocabulary[current_word] += 1
        
    def generate_sentence(self, numWords, smoothing=None, emotion=None, emotion_scores=None):
        '''
        BRIEF:      Generating a sentence of a given word-count and using a given smoothing.
        PARAMETERS: Word-count (numWords) | The smoothing technique to use (smoothing).
        RETURN:     The generated sentence as a string.
        '''
        current_word = self.tokenInitiate
        sentence = []
        
        for _ in range(numWords):
            
            # next_word_candidates = list(self.bigramCounts[current_word].keys())
            next_word_candidates = list(self.vocabulary.keys())
            probabilities = []
            # TODO: Kneser-Ney Smoothing
            if smoothing == 'laplace':
                probabilities = self.__laplace_probabilities(current_word, next_word_candidates, emotion, emotion_scores)
            else:
                probabilities = self.__simple_probabilities(current_word, next_word_candidates, emotion, emotion_scores)
            
            next_word = '<s>'
            while next_word == '<s>':
                next_word = random.choices(next_word_candidates, probabilities)[0]
                        
            if next_word == '</s>':
                break

            sentence.append(next_word)
            current_word = next_word
        
        return " ".join(sentence)
    
    
    # def generate_emotion_samples(self, num_samples_per_emotion=50, smoothing=None, emotion=None):
    #     '''
    #     BRIEF:      Generate emotion-oriented sentences and store them in .txt files.
    #     PARAMETERS: Number of samples per emotion (num_samples_per_emotion) | Smoothing technique to use (smoothing) | List of emotions (emotions).
    #     '''
    #     if emotions is None:
    #         emotions = ['fear', 'anger', 'sadness', 'joy', 'love', 'surprise']

    #     for em in emotion:
    #         samples = []
    #         for _ in range(num_samples_per_emotion):
    #             sentence = self.generate_sentence(numWords=10, smoothing=smoothing)  # Adjust numWords as needed
    #             samples.append(sentence)

    #         filename = f'samples_{em}.txt'
    #         try:
    #             with open(filename, 'w') as file:
    #                 file.write('\n'.join(samples))
    #         except ValueError as e:
    #             print(f"Error writing to file {filename}: {e}")


with open('Assignment1\Task2\\bigram_counts.txt', 'r') as f:
    emotion = f.read().splitlines()
    
with open('Assignment1\data\corpus.txt', 'r') as f:
    corpus = f.read().splitlines()

bigram_data = pd.DataFrame()

bigram_data['word1'] = [i.split('-')[0] for i in emotion]
bigram_data['word2'] = [i.split('-')[1] for i in emotion]
bigram_data['emotion'] = [i.split('-')[2] for i in emotion]

# Creating a bigram model
bigram_model = BigramLM()
bigram_model.fit(corpus)

sentence = ""

try:
    sentence = bigram_model.generate_sentence(10, smoothing=None, emotion='joy', emotion_scores=bigram_data)
except ValueError:
    sentence = bigram_model.generate_sentence(10, smoothing=None, emotion='joy', emotion_scores=bigram_data)

print(sentence)

# # Generating emotion-oriented samples
# if __name__ == "__main__":
#     with open('../data/corpus.txt', 'r') as f:
#         corpus = f.readlines()
#     bigram_model = BigramLM()
#     bigram_model.fit(corpus)
#     emotions = ['happy', 'sadness', 'angry', 'fear', 'love', 'surprise']
#     bigram_model.generate_emotion_samples(num_samples_per_emotion=50, smoothing=None, emotions=emotions)