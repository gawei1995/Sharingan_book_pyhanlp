import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from pyhanlp import PerceptronLexicalAnalyzer, HanLP, CustomDictionary
from ch03.msr import msr_model

HanLP.Config.ShowTermNature = False
segment = PerceptronLexicalAnalyzer(msr_model).enableCustomDictionary(False)
text = "与川普通电话"
print(segment.seg(text))

CustomDictionary.insert("川普", "nrf 1")
segment.enableCustomDictionaryForcing(True)
print(segment.seg(text))

print(segment.seg("银川普通人与川普通电话讲四川普通话"))

segment.enableCustomDictionary(False)
for i in range(3):                                  # 学三遍
    segment.learn("人 与 川普 通电话")                # 在线学习接口的输入必须是标注样本
print(segment.seg("银川普通人与川普通电话讲四川普通话"))
print(segment.seg("首相与川普通话讨论四川普通高考"))