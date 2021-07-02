from tempfile import NamedTemporaryFile

import numpy as np

__doc__ = '试验语料库规模对准确率的影响'

import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from pyhanlp import *
from ch03.ngram_segment import CWSEvaluator
from ch03.msr import msr_train, msr_model, msr_dict, msr_gold, msr_output, msr_test
from ch05.perceptron_cws import CWSTrainer, PerceptronLexicalAnalyzer
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def train_evaluate(ratio):
    partial_corpus = NamedTemporaryFile(delete=False).name
    with open(msr_train, encoding='utf-8') as src, open(partial_corpus, 'w', encoding='utf-8') as dst:
        all_lines = src.readlines()
        dst.writelines(all_lines[:int(ratio * len(all_lines))])

    model = CWSTrainer().train(partial_corpus, partial_corpus, msr_model, 0, 50, 8).getModel()  # 训练模型
    result = CWSEvaluator.evaluate(PerceptronLexicalAnalyzer(model).enableCustomDictionary(False),
                                   msr_test, msr_output, msr_gold, msr_dict)
    # return result.F1
    return float(str(result).split()[2][3:])


if __name__ == '__main__':
    x = [r / 10 for r in range(1, 11)]
    y = [train_evaluate(r) for r in x]
    plt.title("语料库规模对准确率的影响")
    plt.xlabel("语料库规模（万字符）")
    plt.ylabel("准确率")
    plt.xticks([int(r / 10 * 405) for r in range(1, 11)])
    plt.yticks(np.arange(91, 97.5, 0.5))
    plt.plot([int(r / 10 * 405) for r in range(1, 11)], y, color='b')
    plt.grid()
    plt.show()