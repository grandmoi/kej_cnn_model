# -*- coding: utf-8 -*-

import logging

from gensim.models import word2vec

def main():
    # 模型讀取方式
    model = word2vec.Word2Vec.load("../w2vmodel/word2vec.model")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("../语料/UNKNOWN_WORD.txt")
    # model = word2vec.Word2Vec(sentences, size=250,min_count=5)
    #保存模型，供日後使用
    model.build_vocab(sentences,update=True)
    # model.train(sentences)
    model.save("../w2vmodel/word2vec2.model")

    #模型讀取方式
    # model = word2vec.Word2Vec.load("your_model_name")

if __name__ == "__main__":
    main()
