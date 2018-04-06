import codecs

with codecs.open("/home/hunter/语料/abstract2011_1.txt", "r", "utf8") as file:
    lines = file.readlines()

with codecs.open("/home/hunter/语料/abstract2011_1_0-3000.txt","w","utf8") as file:
    file.writelines(lines[:3000])