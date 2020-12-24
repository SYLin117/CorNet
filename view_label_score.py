import numpy as np


def show_score_label():
    score = np.load("./datasets/EUR-Lex/results/CorNetXMLCNN-EUR-Lex-scores.npy")
    label = np.load("./datasets/EUR-Lex/results/CorNetXMLCNN-EUR-Lex-scores.npy")

    print(score)
    print(label)


if __name__ == '__main__':
    show_score_label()
