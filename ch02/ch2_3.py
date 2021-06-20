import sys
sys.path.append(r"C:\Users\gawei\.PyCharmCE2018.3\config\book_pyhanlp")
from pyhanlp import *
from ch02.ch2_2 import load_dictionary
#完全切分算法

def fully_segment(text,dic):
    word_list = []
    for i in range(len(text)):
        for j in range(i+1,len(text)+1):
            word = text[i:j]
            if word in dic:
                word_list.append(word)
    return word_list

def forward_segment(text,dic):
    word_list= []
    i =  0
    while i < len(text):
        longest_word = text[i]
        for j in range(i+1,len(text)+1):
            word = text[i:j]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
        word_list.append(longest_word)
        i += len(longest_word)
    return word_list

def backward_segment(text,dic):
    word_list = []
    i = len(text)-1
    while i >=0:
        longest_word = text[i]
        for j in range(0,i):
            word = text[j:i+1]
            if word in dic:
                if len(word) > len(longest_word):
                    longest_word = word
                    break #越长优先级越高
        word_list.append(longest_word)
        i -= len(longest_word)
    return word_list[::-1]


def count_single_char(word_list:list):
    return sum(1 for word in word_list if len(word)==1)

def bidirectional_segment(text,dic):
    f = forward_segment(text,dic)
    b = backward_segment(text,dic)
    if len(f)<len(b):
        return f
    elif len(f) > len(b):
        return b
    else:
        if count_single_char(f) > count_single_char(b):
            return b
        else:
            return f

if __name__ == '__main__':

    dic = load_dictionary()
    print(fully_segment("商品和服务",dic))

    print(forward_segment("就读北京大学",dic))

    print(backward_segment("就读北京大学",dic))

