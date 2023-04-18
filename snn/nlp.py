import numpy
from typing import List

def tokenize(text:str):
    return text.replace(".", "").lower().split(" ")

def build_vectorized_dict(text:List[str]):
    dct = {}
    i = 0
    for word in text:
        if word not in dct.keys():
            dct[word] = i
            i+=1
    return dct

def vectorize(text:List[str], dct: dict):
    return [dct[word] for word in text]
