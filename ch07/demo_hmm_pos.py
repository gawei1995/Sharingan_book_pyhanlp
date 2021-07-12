import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from  pyhanlp import *
from ch07.pku import PKU199801_TRAIN

HMMPOSTagger = JClass('com.hankcs.hanlp.model.hmm.HMMPOSTagger')
AbstractLexicalAnalyzer = JClass('com.hankcs.hanlp.tokenizer.lexical.AbstractLexicalAnalyzer')
PerceptronSegmenter = JClass('com.hankcs.hanlp.model.perceptron.PerceptronSegmenter')
FirstOrderHiddenMarkovModel = JClass('com.hankcs.hanlp.model.hmm.FirstOrderHiddenMarkovModel')
SecondOrderHiddenMarkovModel = JClass('com.hankcs.hanlp.model.hmm.SecondOrderHiddenMarkovModel')

def train_hmm_pos(corpus, model):
    tagger = HMMPOSTagger(model)  # 创建词性标注器
    tagger.train(corpus)  # 训练
    print(', '.join(tagger.tag("他", "的", "希望", "是", "希望", "上学")))  # 预测
    analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(), tagger)  # 构造词法分析器
    print(analyzer.analyze("他的希望是希望上学"))  # 分词+词性标注
    return tagger


if __name__ == '__main__':
    tagger = train_hmm_pos(PKU199801_TRAIN, FirstOrderHiddenMarkovModel())
    tagger = train_hmm_pos(PKU199801_TRAIN, SecondOrderHiddenMarkovModel())  # 或二阶隐马