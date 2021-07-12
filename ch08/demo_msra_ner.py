
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")

from pyhanlp import *
from ch07.demo_perceptron_pos import POSTrainer
from ch08.demo_hmm_ner import PerceptronSegmenter, PerceptronPOSTagger
from ch08.demo_sp_ner import PerceptronNERecognizer, NERTrainer
from ch08.msra_ner import MSRA_NER_TRAIN


def train_ner(corpus):
    model = os.path.join(os.path.dirname(corpus), 'ner.bin')
    if os.path.isfile(model):
        return PerceptronNERecognizer(model)
    trainer = NERTrainer()
    trainer.tagSet.nerLabels.clear()  # 不识别nr、ns、nt
    trainer.tagSet.nerLabels.addAll(
        ["AGE", "ANGLE", "AREA", "CAPACTITY", "DATE", "DECIMAL", "DURATION", "FRACTION", "FREQUENCY", "INTEGER",
         "LENGTH", "LOCATION", "MEASURE", "MONEY", "ORDINAL", "ORGANIZATION", "PERCENT", "PERSON", "PHONE",
         "POSTALCODE", "RATE", "SPEED", "TEMPERATURE", "TIME", "WEIGHT", "WWW"])
    return PerceptronNERecognizer(trainer.train(corpus, model).getModel())


def train_pos(corpus):
    model = os.path.join(os.path.dirname(corpus), 'pos.bin')
    if os.path.isfile(model):
        return PerceptronPOSTagger(model)
    trainer = POSTrainer()
    return PerceptronPOSTagger(trainer.train(corpus, model).getModel())


if __name__ == '__main__':
    tagger = train_pos(MSRA_NER_TRAIN)
    recognizer = train_ner(MSRA_NER_TRAIN)
    analyzer = PerceptronLexicalAnalyzer(PerceptronSegmenter(), tagger, recognizer)
    analyzer.enableCustomDictionary(False)
    print(analyzer.analyze('2008年5月20日山东大连气温30多摄氏度，王莲香首场赢下李钊颖，中国女队有机会赢下韩国队'))