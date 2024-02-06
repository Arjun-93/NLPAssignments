# Part 1: Byte Pair Encoding

import re
from collections import defaultdict

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S|\$)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_vocab(data):
    vocab = defaultdict(int)
    for line in data:
        for word in line.split():
            token = ' '.join(list(word)) + ' $'
            vocab[token] += 1
    return vocab

def byte_pair_encoding(data, n, vocab_output_file, merge_rule_output_file, tokenized_samples_output_file):
    vocab = get_vocab(data)

    # Write all possible tokens in the vocabulary to the file
    with open(vocab_output_file, 'w') as f:
        for token in vocab:
            f.write(token + '\n')

    for i in range(n):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

        # Write merge rules to the file
        with open(merge_rule_output_file, 'a') as f_merge:
            f_merge.write(','.join(best) + '\n')

        # Write tokens after each iteration to the file
        with open(tokenized_samples_output_file, 'a') as f_samples:
            for token in vocab:
                f_samples.write(token.replace(' ', ',') + '\n')

    return vocab

# read corpus from text file
with open('Assignment1\data\corpus.txt', 'r') as file:
    data = file.read().splitlines()
    
n = 100
vocab_output_file = 'Assignment1\Task1\\all_possible_tokens.txt'
merge_rule_output_file = 'Assignment1\Task1\merge_rules.txt'
tokenized_samples_output_file = 'Assignment1\Task1\\tokenized_samples.txt'

open(vocab_output_file, 'w').close()
open(merge_rule_output_file, 'w').close()
open(tokenized_samples_output_file, 'w').close()

byte_pair_encoding(data, n, vocab_output_file, merge_rule_output_file, tokenized_samples_output_file)