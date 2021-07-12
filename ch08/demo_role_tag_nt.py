
from pyhanlp import *
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch03.ngram_segment import DijkstraSegment
from ch07 import pku
from ch03.demo_corpus_loader import ensure_data,test_data_path


EasyDictionary = JClass('com.hankcs.hanlp.corpus.dictionary.EasyDictionary')
NTDictionaryMaker = JClass('com.hankcs.hanlp.corpus.dictionary.NTDictionaryMaker')
Sentence = JClass('com.hankcs.hanlp.corpus.document.sentence.Sentence')
MODEL = test_data_path() + "/ns"


def train(corpus, model):
    dictionary = EasyDictionary.create(HanLP.Config.CoreDictionaryPath)  # 核心词典
    maker = NTDictionaryMaker(dictionary)  # 训练模块
    maker.train(corpus)  # 在语料库上训练
    maker.saveTxtTo(model)  # 输出HMM到txt


def load(model):
    HanLP.Config.PlaceDictionaryPath = model + ".txt"  # data/test/ns.txt
    HanLP.Config.PlaceDictionaryTrPath = model + ".tr.txt"  # data/test/ns.tr.txt
    segment = DijkstraSegment().enableOrganizationRecognize(True).enableCustomDictionary(False)  # 该分词器便于调试
    return segment


def test(model=MODEL):
    segment = load(model)
    HanLP.Config.enableDebug()
    print(segment.seg("温州黄鹤皮革制造有限公司是由黄先生创办的企业"))


if __name__ == '__main__':
    train(pku.PKU199801, MODEL)
    test(MODEL)