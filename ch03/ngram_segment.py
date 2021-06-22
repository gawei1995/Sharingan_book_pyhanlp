from jpype import *
import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from pyhanlp import *
from ch03.demo_corpus_loader import my_cws_corups,test_data_path,ensure_data
from ch03.msr import msr_model

NatureDictionaryMaker = SafeJClass('com.hankcs.hanlp.corpus.dictionary.NatureDictionaryMaker')
CorpusLoader = SafeJClass('com.hankcs.hanlp.corpus.document.CorpusLoader')
WordNet = JClass('com.hankcs.hanlp.seg.common.WordNet')
Vertex = JClass('com.hankcs.hanlp.seg.common.Vertex')
ViterbiSegment = JClass('com.hankcs.hanlp.seg.Viterbi.ViterbiSegment')
DijkstraSegment = JClass('com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment')
CoreDictionary = LazyLoadingJClass('com.hankcs.hanlp.dictionary.CoreDictionary')
Nature = JClass('com.hankcs.hanlp.corpus.tag.Nature')

def train_bigram(corpus_path,model_path):
    sents = CorpusLoader.convert2SentenceList(corpus_path)
    for sent in sents:
        for word in sent:
            if word.label is None:
                word.setLabel('n')
    maker = NatureDictionaryMaker()
    maker.compute(sents)
    maker.saveTxtTo(model_path)

def load_bigram(model_path,verbose = True,ret_viterbi=True):
    HanLP.Config.CoreDictionaryPath = model_path + ".txt"  # unigram
    HanLP.Config.BiGramDictionaryPath = model_path + ".ngram.txt"  # bigram

    HanLP.Config.CoreDictionaryTransformMatrixDictionaryPath = model_path + ".tr.txt"  # 词性转移矩阵，分词时可忽略

    if model_path != msr_model:
        with open(HanLP.Config.CoreDictionaryTransformMatrixDictionaryPath, encoding='utf-8') as src:
            for tag in src.readline().strip().split(",")[1:]:
                Nature.create(tag)
    CoreBiGramTableDictionary = SafeJClass('com.hankcs.hanlp.dictionary.CoreBiGramTableDictionary')
    CoreDictionary.getTermFrequency("商品")

    if verbose:
        print(CoreDictionary.getTermFrequency("商品"))
        print(CoreBiGramTableDictionary.getBiFrequency("商品", "和"))
        sent = '商品和服务'
        # sent = '货币和服务'
        wordnet = generate_wordnet(sent, CoreDictionary.trie)
        print(wordnet)
        print(viterbi(wordnet))
    return ViterbiSegment().enableAllNamedEntityRecognize(False).enableCustomDictionary(
        False) if ret_viterbi else DijkstraSegment().enableAllNamedEntityRecognize(False).enableCustomDictionary(False)

def generate_wordnet(sent,trie):
    """
    生成词网
    :param sent:句子
    :param trie:词典
    :return:词网
    """

    searcher = trie.getSearcher(JString(sent),0)
    wordnet = WordNet(sent)
    while searcher.next():
        wordnet.add(searcher.begin + 1,
                    Vertex(sent[searcher.begin:searcher.begin+searcher.length],searcher.value,searcher.index))
    #原子分词，保持图联通
    vertexes = wordnet.getVertexes()
    i = 0
    while i<len(vertexes):
        if len(vertexes[i]) == 0:
            j = i+1
            for j in range(i+1,len(vertexes)-1):
                if len(vertexes[j]):
                    break
            wordnet.add(i,Vertex.newPunctuationInstance(sent[i-1:j-1]))
            i = j
        else:
            i += len(vertexes[i][-1].realWord)
    return wordnet

def viterbi(wordnet):
    nodes = wordnet.getVertexes()
    for i in range(0,len(nodes)-1):
        for node in nodes[i]:
            for to in nodes[i+len(node.realWord)]:
                to.updateFrom(node)

    path =[]
    f = nodes[len(nodes)-1].getFirst()
    while f:
        path.insert(0,f)
        f = f.getFrom()

    return [v.realWord for v in path]

if __name__ == '__main__':

    corpus_path = my_cws_corups()
    model_path = os.path.join(test_data_path(),'my_cws_model')
    train_bigram(corpus_path,model_path)
    load_bigram(model_path)

    ViterbiSegment = SafeJClass('com.hankcs.hanlp.seg.Viterbi.ViterbiSegment')

    segment = ViterbiSegment()
    sentence = "社会摇摆简称社会摇"
    segment.enableCustomDictionary(False)
    print("不挂载词典：", segment.seg(sentence))
    CustomDictionary.insert("社会摇", "nz 100")
    segment.enableCustomDictionary(True)
    print("低优先级词典：", segment.seg(sentence))
    segment.enableCustomDictionaryForcing(True)
    print("高优先级词典：", segment.seg(sentence))

    segment = load_bigram(model_path=msr_model, verbose=False, ret_viterbi=False)
    assert CoreDictionary.contains("管道")
    text = "北京输气管道工程"
    HanLP.Config.enableDebug()
    print(segment.seg(text))

    HanLP.Config.BiGramDictionaryPath = msr_model + ".ngram.txt"
    print(HanLP.Config.BiGramDictionaryPath)


    jp_corpus = ensure_data('jpcorpus',
                            'http://file.hankcs.com/corpus/jpcorpus.zip')
    jp_bigram = os.path.join(jp_corpus, 'jp_bigram')
    jp_corpus = os.path.join(jp_corpus, 'ja_gsd-ud-train.txt')

    train_bigram(jp_corpus, jp_bigram)  # 训练
    segment = load_bigram(jp_bigram, verbose=False)  # 加载
    print(segment.seg('自然言語処理入門という本が面白いぞ！'))  # 日语分词