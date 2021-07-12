from pyhanlp import *
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch07 import pku
from ch08.demo_hmm_ner import test

NERTrainer = JClass('com.hankcs.hanlp.model.perceptron.NERTrainer')
CRFNERecognizer = JClass('com.hankcs.hanlp.model.crf.CRFNERecognizer')


def train(corpus, model):
    recognizer = CRFNERecognizer(None)  # 空白
    recognizer.train(corpus, model)
    recognizer = CRFNERecognizer(model)
    return recognizer


if __name__ == '__main__':
    recognizer = train(pku.PKU199801_TRAIN, pku.NER_MODEL)
    test(recognizer)