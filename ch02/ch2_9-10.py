import re
from pyhanlp import *

import zipfile
import os

from pyhanlp.static import download, remove_file, HANLP_DATA_PATH


def test_data_path():
    """
    获取测试数据路径，位于$root/data/test，根目录由配置文件指定。
    :return:
    """
    data_path = os.path.join(HANLP_DATA_PATH, 'test')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    return data_path


def ensure_data(data_name, data_url):
    root_path = test_data_path()
    dest_path = os.path.join(root_path, data_name)
    if os.path.exists(dest_path):
        return dest_path
    if data_url.endswith('.zip'):
        dest_path += '.zip'
    download(data_url, dest_path)
    if data_url.endswith('.zip'):
        with zipfile.ZipFile(dest_path, "r") as archive:
            archive.extractall(root_path)
        remove_file(dest_path)
        dest_path = dest_path[:-len('.zip')]
    return dest_path
def to_region(segmentation:str) -> list:
    """
    分词转换成区间
    :param segmentation:商品和服务
    :return:[(0,2),(2,3),(3,5)]
    """
    region = []
    start = 0
    for word in re.compile("\\s+").split(segmentation.strip()):
        end = start + len(word)
        region.append((start,end))
        start = end
    return region

def prf(gold:str,pred:str,dic) -> tuple:
    """
    计算P,R,F1
    :param gold:标准答案文件
    :param pred:分词结果文件
    :param dic:词典
    :return:（P,R,F1,OOV_R,IV_R）
    """
    A_size, B_size, A_cap_B_size, OOV, IV, OOV_R, IV_R = 0, 0, 0, 0, 0, 0, 0
    with open(gold, encoding='utf-8') as gd, open(pred, encoding='utf-8') as pd:
        for g, p in zip(gd, pd):
            A, B = set(to_region(g)), set(to_region(p))
            A_size += len(A)
            B_size += len(B)
            A_cap_B_size += len(A & B)
            text = re.sub("\\s+", "", g)
            for (start, end) in A:
                word = text[start: end]
                if dic.containsKey(word):
                    IV += 1
                else:
                    OOV += 1

            for (start, end) in A & B:
                word = text[start: end]
                if dic.containsKey(word):
                    IV_R += 1
                else:
                    OOV_R += 1
    p, r = A_cap_B_size / B_size * 100, A_cap_B_size / A_size * 100
    return p, r, 2 * p * r / (p + r), OOV_R / OOV * 100, IV_R / IV * 100

from jpype import JString
from pyhanlp import *

def load_from_file(path):
    """
    从词典文件加载DoubleArrayTrie
    :param path:词典路径
    :return:双数组trie树
    """
    map=JClass('java.util.TreeMap')()
    with open(path,encoding='utf-8') as src:
        for word in src:
            word = word.strip()
            map[word] = word
    return  JClass('com.hankcs.hanlp.collection.trie.DoubleArrayTrie')(map)

def load_from_words(*words):
    """
    从词汇构造双数组trie树
    :param words: 一系列词语
    :return:
    """
    map = JClass('java.util.TreeMap')()  # 创建TreeMap实例
    for word in words:
        map[word] = word
    return JClass('com.hankcs.hanlp.collection.trie.DoubleArrayTrie')(map)

def remove_stopwords_termlist(termlist, trie):
    return [term.word for term in termlist if not trie.containsKey(term.word)]


def replace_stropwords_text(text, replacement, trie):
    searcher = trie.getLongestSearcher(JString(text), 0)
    offset = 0
    result = ''
    while searcher.next():
        begin = searcher.begin
        end = begin + searcher.length
        if begin > offset:
            result += text[offset: begin]
            print(text[offset:begin])
        result += replacement
        print(result)
        offset = end
    if offset < len(text):
        result += text[offset:]
    return result

def demo_traditional_chinese2simplified_chinese():
    """ 将简繁转换做到极致
    >>> demo_traditional_chinese2simplified_chinese()
    「以後等你當上皇后，就能買草莓慶祝了」。發現一根白頭髮
    凭借笔记本电脑写程序HanLP
    hankcs在臺灣寫程式碼
    hankcs在台湾写代码
    hankcs在香港寫代碼
    hankcs在香港写代码
    hankcs在臺灣寫程式碼
    hankcs在香港寫代碼
    hankcs在臺灣寫程式碼
    hankcs在台灣寫代碼
    hankcs在臺灣寫代碼
    hankcs在臺灣寫代碼
    """
    print(HanLP.convertToTraditionalChinese("“以后等你当上皇后，就能买草莓庆祝了”。发现一根白头发"))
    print(HanLP.convertToSimplifiedChinese("憑藉筆記簿型電腦寫程式HanLP"))
    # 简体转台湾繁体
    print(HanLP.s2tw("hankcs在台湾写代码"))
    # 台湾繁体转简体
    print(HanLP.tw2s("hankcs在臺灣寫程式碼"))
    # 简体转香港繁体
    print(HanLP.s2hk("hankcs在香港写代码"))
    # 香港繁体转简体
    print(HanLP.hk2s("hankcs在香港寫代碼"))
    # 香港繁体转台湾繁体
    print(HanLP.hk2tw("hankcs在臺灣寫代碼"))
    # 台湾繁体转香港繁体
    print(HanLP.tw2hk("hankcs在香港寫程式碼"))

    # 香港/台湾繁体和HanLP标准繁体的互转
    print(HanLP.t2tw("hankcs在臺灣寫代碼"))
    print(HanLP.t2hk("hankcs在臺灣寫代碼"))

    print(HanLP.tw2t("hankcs在臺灣寫程式碼"))
    print(HanLP.hk2t("hankcs在台灣寫代碼"))


def demo_pinyin_to_chinese():
    """ HanLP中的数据结构和接口是灵活的，组合这些接口，可以自己创造新功能
    >>> demo_pinyin_to_chinese()
    [renmenrenweiyalujiangbujian/null, lvse/[滤色, 绿色]]
    """
    AhoCorasickDoubleArrayTrie = JClass(
        "com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie")
    StringDictionary = JClass(
        "com.hankcs.hanlp.corpus.dictionary.StringDictionary")
    CommonAhoCorasickDoubleArrayTrieSegment = JClass(
        "com.hankcs.hanlp.seg.Other.CommonAhoCorasickDoubleArrayTrieSegment")
    CommonAhoCorasickSegmentUtil = JClass(
        "com.hankcs.hanlp.seg.Other.CommonAhoCorasickSegmentUtil")
    Config = JClass("com.hankcs.hanlp.HanLP$Config")

    TreeMap = JClass("java.util.TreeMap")
    TreeSet = JClass("java.util.TreeSet")

    dictionary = StringDictionary()
    dictionary.load(Config.PinyinDictionaryPath)
    entry = {}
    m_map = TreeMap()
    for entry in dictionary.entrySet():
        pinyins = entry.getValue().replace("[\\d,]", "")
        words = m_map.get(pinyins)
        if words is None:
            words = TreeSet()
            m_map.put(pinyins, words)
        words.add(entry.getKey())
    words = TreeSet()
    words.add("绿色")
    words.add("滤色")
    m_map.put("lvse", words)

    segment = CommonAhoCorasickDoubleArrayTrieSegment(m_map)
    print(segment.segment("renmenrenweiyalujiangbujianlvse"))


def demo_pinyin():
    """ 汉字转拼音
    >>> demo_pinyin()
    原文， 重载不是重任！
    拼音（数字音调）， [chong2, zai3, bu2, shi4, zhong4, ren4, none5]
    拼音（符号音调）， chóng, zǎi, bú, shì, zhòng, rèn, none,
    拼音（无音调）， chong, zai, bu, shi, zhong, ren, none,
    声调， 2, 3, 2, 4, 4, 4, 5,
    声母， ch, z, b, sh, zh, r, none,
    韵母， ong, ai, u, i, ong, en, none,
    输入法头， ch, z, b, sh, zh, r, none,
    jie zhi none none none none nian none
    jie zhi 2 0 1 2 nian ，
    """
    Pinyin = JClass("com.hankcs.hanlp.dictionary.py.Pinyin")
    text = "重载不是重任！"
    pinyin_list = HanLP.convertToPinyinList(text)

    print("原文，", end=" ")
    print(text)
    print("拼音（数字音调），", end=" ")
    print(pinyin_list)
    print("拼音（符号音调），", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getPinyinWithToneMark(), end=" ")
    print("\n拼音（无音调），", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getPinyinWithoutTone(), end=" ")
    print("\n声调，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getTone(), end=" ")
    print("\n声母，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getShengmu(), end=" ")
    print("\n韵母，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getYunmu(), end=" ")
    print("\n输入法头，", end=" ")
    for pinyin in pinyin_list:
        print("%s," % pinyin.getHead(), end=" ")

    print()
    # 拼音转换可选保留无拼音的原字符
    print(HanLP.convertToPinyinString("截至2012年，", " ", True))
    print(HanLP.convertToPinyinString("截至2012年，", " ", False))
if __name__ == '__main__':
    print(to_region('商品 和 服务'))

    sighan05 = ensure_data('icwb2-data', 'http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip')
    msr_dict = os.path.join(sighan05, 'gold', 'msr_training_words.utf8')
    msr_test = os.path.join(sighan05, 'testing', 'msr_test.utf8')
    msr_output = os.path.join(sighan05, 'testing', 'msr_output.txt')
    msr_gold = os.path.join(sighan05, 'gold', 'msr_test_gold.utf8')

    DoubleArrayTrieSegment = JClass('com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment')
    segment = DoubleArrayTrieSegment([msr_dict]).enablePartOfSpeechTagging(True)
    with open(msr_gold, encoding='utf-8') as test, open(msr_output, 'w', encoding='utf-8') as output:
        for line in test:
            output.write("  ".join(term.word for term in segment.seg(re.sub("\\s+", "", line))))
            output.write("\n")
    print("P:%.2f R:%.2f F1:%.2f OOV-R:%.2f IV-R:%.2f" % prf(msr_gold, msr_output, segment.trie))

    HanLP.Config.ShowTermNature = False
    trie = load_from_file(HanLP.Config.CoreStopWordDictionaryPath)
    text = "停用词的意义相对而言无关紧要吧。"
    segment = DoubleArrayTrieSegment()
    termlist = segment.seg(text)
    print("分词结果：", termlist)
    print("分词结果去除停用词：", remove_stopwords_termlist(termlist, trie))
    trie = load_from_words("的", "相对而言", "吧")
    print("不分词去掉停用词", replace_stropwords_text(text, "**", trie))

    import doctest
    doctest.testmod(verbose=True)

    doctest.testmod(verbose=True)

    doctest.testmod(verbose=True, optionflags=doctest.NORMALIZE_WHITESPACE)