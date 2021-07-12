from pyhanlp import *
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch07 import pku
from ch08.demo_hmm_ner import test, PerceptronSegmenter, PerceptronPOSTagger
from ch08.demo_role_tag_nr import Sentence

NERTrainer = JClass('com.hankcs.hanlp.model.perceptron.NERTrainer')
PerceptronNERecognizer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronNERecognizer')


def train(corpus, model):
    trainer = NERTrainer()
    return PerceptronNERecognizer(trainer.train(corpus, model).getModel())


if __name__ == '__main__':
    recognizer = train(pku.PKU199801_TRAIN, pku.NER_MODEL)
    test(recognizer)
    analyzer = PerceptronLexicalAnalyzer(PerceptronSegmenter(), PerceptronPOSTagger(), recognizer)  # ①
    analyzer.enableCustomDictionary(False)
    sentence = Sentence.create("与/c 特朗普/nr 通/v 电话/n 讨论/v [太空/s 探索/vn 技术/n 公司/n]/nt")  # ②
    while not analyzer.analyze(sentence.text()).equals(sentence):  # ③
        analyzer.learn(sentence)