import spacy
from spacy import displacy


def printRelationsDependencies():
    print(f"{'Word':<10} {'Lemme':<10} {'pos':<10} {'depenedency':<15} {'head':<15}")
    for token in text:
        print(f"{token.text:<10} {token.lemma:<10} {token.pos_:<10} {token.dep_:<15} {token.head.text:<15}")

def display():
    displacy.serve(text, style="dep")

if __name__=="__main__":

    english = spacy.load("en_core_web_sm")
    text = english("The quick brown fox jumps over the lazy dog.")
    printRelationsDependencies()
    display()