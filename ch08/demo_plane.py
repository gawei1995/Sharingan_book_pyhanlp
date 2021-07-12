
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")

from ch05.perceptron_cws import CWSTrainer
from ch07.demo_hmm_pos import AbstractLexicalAnalyzer, PerceptronSegmenter
from ch07.demo_perceptron_pos import PerceptronPOSTagger
from ch08.demo_sp_ner import NERTrainer, os, PerceptronNERecognizer
from ch03.demo_corpus_loader import ensure_data,test_data_path

PLANE_ROOT = ensure_data("plane-re", "http://file.hankcs.com/corpus/plane-re.zip")
PLANE_CORPUS = os.path.join(PLANE_ROOT, 'train.txt')
PLANE_MODEL = os.path.join(PLANE_ROOT, 'model.bin')

if __name__ == '__main__':
    trainer = NERTrainer()
    trainer.tagSet.nerLabels.clear()  # 不识别nr、ns、nt
    trainer.tagSet.nerLabels.add("np")  # 目标是识别np
    recognizer = PerceptronNERecognizer(trainer.train(PLANE_CORPUS, PLANE_MODEL).getModel())
    # 在NER预测前，需要一个分词器，最好训练自同源语料库
    CWS_MODEL = CWSTrainer().train(PLANE_CORPUS, PLANE_MODEL.replace('model.bin', 'cws.bin')).getModel()
    analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(CWS_MODEL), PerceptronPOSTagger(), recognizer)
    print(analyzer.analyze("米高扬设计米格-17PF：米格-17PF型战斗机比米格-17P性能更好。"))
    print(analyzer.analyze("米格-阿帕奇-666S横空出世。"))