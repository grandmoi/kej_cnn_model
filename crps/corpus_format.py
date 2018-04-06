import codecs
import jieba
import jieba.posseg as pseg
import random

class CorpusFormat(object):

    relation_table = {"cs-ef":0,"ef-cs":1,
                      "identity":2,
                      "way-obj":3,"obj-way":4,
                      "en-or":5,"or-en":6,
                      "other":7,
                      "loc-aA":8,"loc-Aa":9,
                      "clsaA":10,"clsAa":11,
                      "w-c":12,"c-w":13,
                      "related":14,
                      "pdr-pdt":15,"pdt-pdr":16,
                      "ag-ins":17,"ins-ag":18,
                      "med-ill":19,"ill-med":20,
                      "top-msg":21,"msg-top":21
                      }

    def freq_tag(self,word):
        freq = jieba.get_FREQ(word)
        tag = ""
        if freq is not None:
            tag = pseg.lcut(word,HMM=False)[0].flag
        return freq,tag

    def recover_dict(self,word,freq,tag):
        if freq is None:
            jieba.del_word(word)
        else:
            jieba.add_word(word, freq=freq, tag=tag)

    def word_divide(self,inputfiles,outputfile):
        outputs = []
        relations = {}
        for file in inputfiles:
            with codecs.open(file,"r","utf8") as read:
                lines = read.readlines()
                for line in lines:
                    if line == "\n":
                        continue
                    line = line.replace("<Entity>", "")
                    line = line.replace("</Entity>", "")
                    line = line.replace("<kej>", "")
                    line = line.replace("</kej>", "")
                    line = line.replace(" ","").strip()
                    temps = line.split("\t")
                    if len(temps)!=4:
                        print("wrong")
                    relation = temps[0]
                    entity1 = temps[1]
                    freq1,tag1 = self.freq_tag(entity1)
                    jieba.add_word(entity1,freq = 1000000,tag = "kej")
                    entity2 = temps[2]
                    freq2, tag2 = self.freq_tag(entity2)
                    jieba.add_word(entity2, freq=1000000, tag="kej")
                    words = jieba.lcut(temps[3])
                    sentence = ""
                    # sentence_label = ""
                    seq1 = []
                    seq2 = []
                    pos1 = ""
                    pos2 = ""
                    for word in words:
                        if word == " ":
                            continue
                        sentence += word
                        sentence += " "
                    sentence = sentence.strip()
                    words = sentence.split(" ")
                    for i in range(len(words)):
                        if words[i] == entity1:
                            seq1.append(i)
                        if words[i] == entity2:
                            seq2.append(i)
                    for item in seq1:
                        pos1 += (str(item) + " ")
                    for item in seq2:
                        pos2 += (str(item) + " ")
                    pos1 = pos1.strip()
                    pos2 = pos2.strip()
                    sentence = sentence
                    # sentence_label = sentence_label.strip()
                    if pos1 == "" or pos2 == "":
                        print("分词错误" + entity1,entity2,sentence)
                    else:
                        if relation not in relations:
                            relations[relation] = 1
                        else:
                            relations[relation] += 1
                        outputs.append(relation + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    self.recover_dict(entity1,freq1,tag1)
                    self.recover_dict(entity2, freq2, tag2)
        print(relations)
        with codecs.open(outputfile,"w","utf8") as file:
            file.writelines(outputs)


    def change_order(self,inputfile,outputfile):
        outputs = []
        with codecs .open(inputfile,"r","utf8") as file:
            lines = file.readlines()
            for line in lines:
                temps = line.split("\t")
                if len(temps) != 4:
                    print("wrong")
                relation = temps[0]
                pos1 = temps[1]
                pos2 = temps[2]
                sentence = temps[3].strip()
                if relation == "cause-effect":
                    outputs.append("cs-ef" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("ef-cs" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "cs-ef":
                    outputs.append("cs-ef" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("ef-cs" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "classify":
                    outputs.append("clsaA" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("clsAa" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "identity":
                    outputs.append("idnetity" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                elif relation == "way-obj":
                    outputs.append("way-obj" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("obj-way" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "en-or":
                    outputs.append("en-or" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("or-en" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "other":
                    outputs.append("other" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("other" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "location":
                    outputs.append("locaA" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("locAa" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "w-c":
                    outputs.append("w-c" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("c-w" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "related":
                    outputs.append("related" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                elif relation == "prdcr-prdct":
                    outputs.append("pdr-pdt" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("pdt-pdr" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "ag-ins":
                    outputs.append("ag-ins" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("ins-ag" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "med-ill":
                    outputs.append("med-ill" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("ill-med" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
                elif relation == "top-msg":
                    outputs.append("top-msg" + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n")
                    outputs.append("msg-top" + "\t" + pos2 + "\t" + pos1 + "\t" + sentence + "\n")
        with codecs.open(outputfile,"w","utf8") as file:
            file.writelines(outputs)
    def merge_files(self,files,outputfile):
        outputs = []
        for file in files:
            with codecs.open(file,"r","utf8") as read:
                lines = read.readlines()
                outputs.extend(lines)
        print(len(outputs))
        with codecs.open(outputfile,"w","utf8") as file:
            file.writelines(outputs)

    def extract_relation_type(self,inputfile,outputfile):
        relations = {}
        outputs = []
        with codecs.open(inputfile, "r", "utf8") as read:
            lines = read.readlines()
            for line in lines:
                temps = line.split("\t")
                if len(temps) != 4:
                    print("wrong")
                relation = temps[0]
                pos1 = temps[1]
                pos2 = temps[2]
                sentence = temps[3].strip()
                outputs.append([relation,relation + "\t" + pos1 + "\t" + pos2 + "\t" + sentence + "\n"])
                if relation not in relations:
                    relations[relation] = 1
                else:
                    relations[relation] += 1
        output_items = []
        sentences = []
        for key,value in relations.items():
            if value>20:
                output_items.append(key)
        for item in outputs:
            if item[0] in output_items:
                sentences.append(item[1])
            else:
                print(item[0])
        print(len(sentences))
        with codecs.open(outputfile,"w","utf8") as file:
            file.writelines(sentences)
    def train_test(self,files,trainfile,testfile):
        trains = []
        tests = []
        for file in files:
            with codecs.open(file, "r", "utf8") as read:
                lines = read.readlines()
                total = len(lines)
                print(total)
                test = random.sample(lines, int(total / 4))
                for item in test:
                    lines.remove(item)
                print(len(test))
                print(len(lines))
                trains.extend(lines)
                tests.extend(test)
        print(str(len(trains)) + " " + str(len(tests)))
        with codecs.open(trainfile, "w", "utf8") as wrt:
            wrt.writelines(trains)
        with codecs.open(testfile, "w", "utf8") as wrt:
            wrt.writelines(tests)

cf = CorpusFormat()
# cf.word_divide(["../训练语料/zhiliao2.txt"],"../训练语料/zhiliao2_outputs.txt")
cf.train_test(["train_all2.txt"],"../files/train.txt","../files/test.txt")
# cf.word_divide(["/home/hunter/PycharmProjects/corpus_extraction/处理后语料/部分整体.txt"],"./output_whole.txt")
# cf.change_order("../训练语料/zhiliao2_outputs.txt","../训练语料/zhiliao2_divides.txt")
# cf.merge_files(["divides.txt","divide_cause.txt","divide_classify.txt","divide_whole.txt","divide_zhiliao.txt","train_locAa.txt","train_locaA.txt","train_other.txt"],"samples_all.txt")
# cf.extract_relation_type("samples_all.txt","train_all2.txt")
# cf.merge_files(["../训练语料/zhiliao2_divides.txt","../训练语料/train_all.txt"],"samples_all.txt")