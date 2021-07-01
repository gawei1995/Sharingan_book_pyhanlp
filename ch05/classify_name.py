import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from pyhanlp import *
from ch03.demo_corpus_loader import my_cws_corups,test_data_path,ensure_data

PerceptronNameGenderClassifier = JClass('com.hankcs.hanlp.model.perceptron.PerceptronNameGenderClassifier')
cnname = ensure_data('cnname', 'http://file.hankcs.com/corpus/cnname.zip')
TRAINING_SET = os.path.join(cnname, 'train.csv')
TESTING_SET = os.path.join(cnname, 'test.csv')
MODEL = cnname + ".bin"

def run_classifier(averaged_perceptron):
    print('=====%s=====' % ('平均感知机算法' if averaged_perceptron else '朴素感知机算法'))
    classifier = PerceptronNameGenderClassifier()
    print('训练集准确率：', classifier.train(TRAINING_SET, 10, averaged_perceptron))
    model = classifier.getModel()
    print("特征数量：",len(model.parameter))
    #model.save(MODEL,model.featureMap.entrySet(),0,True)
    #classifier = PerceptronNameGenderClassifier(MODEL)
    for name in "赵建军","沈雁冰","陆雪琪","李冰冰":
        print('%s=%s' % (name, classifier.predict(name)))
    print('测试集准确率：', classifier.evaluate(TESTING_SET))

if __name__ == '__main__':

    run_classifier(True)
    run_classifier(False)