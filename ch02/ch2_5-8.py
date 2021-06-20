from pyhanlp import *

from pyhanlp.static import HANLP_DATA_PATH

def classic_demo():
    words = ["hers","his","she","he"]
    Trie = JClass('com.hankcs.hanlp.algorithm.ahocorasick.trie.Trie')
    trie = Trie()
    for w in words:
        trie.addKeyword(w)

    for emit in trie.parseText("ushers"):
        print("[%d:%d]=%s"%(emit.getStart(),emit.getEnd(),emit.getKeyword()))


def clasic_demo():
    words = ["hers", "his", "she", "he"]
    map = JClass('java.util.TreeMap')()
    for word in words:
        map[word] = word.upper()

    trie = JClass('com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie')(map)
    for hit in trie.parseText("ushers"):
        print("[%d:%d]=%s"%(hit.begin,hit.end,hit.value))
if __name__ == '__main__':
    classic_demo()
    clasic_demo()

    HanLP.Config.ShowTermNature = True
    segment = JClass('com.hankcs.hanlp.seg.Other.AhoCorasickDoubleArrayTrieSegment')(HanLP.Config.CoreDictionaryPath)
    segment = DoubleArrayTrieSegment()
    print(segment.seg("江西鄱阳湖干枯，中国最大淡水湖变成大草原"))
z
    HanLP.Config.ShowTermNature = False
    dict1 = HANLP_DATA_PATH + "/dictionary/CoreNatureDictionary.mini.txt"
    dict2 = HANLP_DATA_PATH + "/dictionary/custom/上海地名.txt ns"
    segment = DoubleArrayTrieSegment(dict1)
    print(segment.seg('江西鄱阳湖干枯，中国最大淡水湖变成大草原'))

    segment = DoubleArrayTrieSegment([dict1, dict2])
    print(segment.seg('上海市虹口区大连西路550号SISU'))

    segment.enablePartOfSpeechTagging(True)
    HanLP.Config.ShowTermNature = True
    print(segment.seg('上海市虹口区大连西路550号SISU'))

    for term in segment.seg('上海市虹口区大连西路550号SISU'):
        print("单词:%s 词性:%s" % (term.word, term.nature))