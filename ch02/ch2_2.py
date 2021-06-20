from pyhanlp import *

def load_dictionary():
    """
    加载hanlp中的mini词库
    :return:一个set的库
    """
    IOUtil = JClass('com.hankcs.hanlp.corpus.io.IOUtil')
    path = HanLP.Config.CoreDictionaryPath.replace('.txt', '.mini.txt')
    dic = IOUtil.loadDictionary([path])
    return set(dic.keySet())

if __name__ == '__main__':
    dic = load_dictionary()
    print(len(dic))
    print(list(dic)[0])



# for term in HanLP.segment("下雨天地面积水"):
#     print("{}\t{}".format(term.word,term.nature))
#
# print(HanLP.segment("你好"))
#
# document = "“对我们共产党人来说，中国革命历史是最好的营养剂。”党的十八大以来，在一些重大历史事件纪念日，习近平总书记都会参观相关主题展览。军事博物馆、国家博物馆、北京展览馆、香山革命纪念馆等，都留下了他的足迹。6月18日，党的百年华诞前夕，习近平来到了新近落成的中国共产党历史展览馆。"
#
# print(HanLP.extractKeyword(document,3))
# print(HanLP.extractKeyword(document,5))
#
# print(HanLP.extractSummary(document,1))
#
# print(HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。"))