import os
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch03.demo_corpus_loader import ensure_data
from ch07.demo_hmm_pos import AbstractLexicalAnalyzer, PerceptronSegmenter
from ch07.demo_perceptron_pos import train_perceptron_pos

ZHUXIAN = ensure_data("zhuxian", "http://file.hankcs.com/corpus/zhuxian.zip") + "/train.txt"
posTagger = train_perceptron_pos(ZHUXIAN)  # 训练
analyzer = AbstractLexicalAnalyzer(PerceptronSegmenter(), posTagger)  # 包装
print(analyzer.analyze("陆雪琪的天琊神剑不做丝毫退避，直冲而上，瞬间，这两道奇光异宝撞到了一起。"))  # 分词+标注