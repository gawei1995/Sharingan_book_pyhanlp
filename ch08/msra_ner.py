import os
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch03.demo_corpus_loader import ensure_data

MSRA_NER = ensure_data("msra-ne", "http://file.hankcs.com/corpus/msra-ne.zip")
MSRA_NER_TRAIN = os.path.join(MSRA_NER, 'train.txt')
MSRA_NER_TEST = os.path.join(MSRA_NER, 'test.txt')
MSRA_NER_MODEL = os.path.join(MSRA_NER, 'model.bin')