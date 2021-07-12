from pyhanlp import *
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from ch03.ngram_segment import ViterbiSegment

CharType = JClass('com.hankcs.hanlp.dictionary.other.CharType')

segment = ViterbiSegment()
print(segment.seg("牛奶三〇〇克壹佰块"))
print(segment.seg("牛奶300克100块"))
print(segment.seg("牛奶300g100rmb"))
# 演示自定义字符类型
text = "牛奶300~400g100rmb"
print(segment.seg(text))
CharType.set('~', CharType.CT_NUM)
print(segment.seg(text))