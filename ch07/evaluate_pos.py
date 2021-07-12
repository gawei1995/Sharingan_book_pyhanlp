from pyhanlp import *
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch07.demo_crf_pos import train_crf_pos
from ch07.demo_hmm_pos import train_hmm_pos, FirstOrderHiddenMarkovModel, SecondOrderHiddenMarkovModel
from ch07.demo_perceptron_pos import train_perceptron_pos
from ch07.pku import PKU199801_TRAIN, PKU199801_TEST

PosTagUtil = JClass('com.hankcs.hanlp.dependency.nnparser.util.PosTagUtil')

print("一阶HMM\t%.2f%%" % (
    PosTagUtil.evaluate(train_hmm_pos(PKU199801_TRAIN, FirstOrderHiddenMarkovModel()), PKU199801_TEST)))
print("二阶HMM\t%.2f%%" % (
    PosTagUtil.evaluate(train_hmm_pos(PKU199801_TRAIN, SecondOrderHiddenMarkovModel()), PKU199801_TEST)))
print("感知机\t%.2f%%" % (PosTagUtil.evaluate(train_perceptron_pos(PKU199801_TRAIN), PKU199801_TEST)))
print("CRF\t%.2f%%" % (PosTagUtil.evaluate(train_crf_pos(PKU199801_TRAIN), PKU199801_TEST)))