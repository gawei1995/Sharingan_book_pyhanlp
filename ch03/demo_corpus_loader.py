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

CorpusLoader = SafeJClass('com.hankcs.hanlp.corpus.document.CorpusLoader')

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

def my_cws_corups():
    data_root = test_data_path()
    corpus_path = os.path.join(data_root,'my_cws_corpus.txt')
    if not os.path.isfile(corpus_path):
        with open(corpus_path,'w',encoding="utf-8") as out:
            out.write('''商品 和 服务
商品 和服 物美价廉
服务 和 货币''')
    return corpus_path

def load_cws_corpus(corpus_path):
    return CorpusLoader.convert2SentenceList(corpus_path)

if __name__ == '__main__':
    corpus_path =my_cws_corups()
    sents = load_cws_corpus(corpus_path)
    for sent in sents:
        print(sent)