import numpy as np
from snn.layers import Dense, Embedding
from snn.loss import mse
from snn.nlp import tokenize, build_vectorized_dict, vectorize
from snn.nn import NN


s = """The AIM-9X is an advanced short-range air-to-air missile used by various military forces around the world. It is designed to engage and destroy targets at close range, making it an ideal weapon for dogfighting scenarios. The missile incorporates several advanced technologies, such as infrared guidance and thrust vectoring, that enhance its accuracy and maneuverability. The AIM-9X has a range of up to 22 miles and can reach speeds of Mach 2.5. It is highly versatile and can be deployed from a variety of aircraft, including fighter jets and helicopters. The missile has undergone several upgrades since its introduction in the early 2000s, ensuring that it remains a highly effective weapon system for modern air warfare."""
tok = tokenize(s)
dct = build_vectorized_dict(tok)
def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

# https://jaketae.github.io/study/word2vec/#generating-training-data
def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i), 
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
    
    return np.asarray(X), np.asarray(y)



l = generate_training_data(tok, dct,2)
print(len(dct))
# layer = Embedding(100,83)
word2vec = NN([Embedding(100,82),Dense(90,100), Dense(82,90)],loss = mse(),kaiming=True, optimizer="rmsprop")

word2vec.train(2000,l[0], l[1])
NN.save("out.pkl",word2vec)
# nn = NN.load("out.pkl")
print(word2vec.forward_layer(l[0].T, 0)[0])
print(tok[0])

