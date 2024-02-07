from A1_2_3_2020184_2020176 import BigramLM

with open('data/corpus.txt', 'r') as f:
    corpus = f.readlines()
bigram_model = BigramLM()
bigram_model.fit(corpus)
emotions = ['happy', 'sadness', 'angry', 'fear', 'love', 'surprise']
bigram_model.generate_emotion_samples(num_samples_per_emotion=50, smoothing=None, emotions=emotions)