from cnn_model import KejModel
from data_tools import DataTools
from ptr.preprocess import  PreTrain

# prt = PreTrain("w2vmodel/word2vec2.model")
# line = "w-c	2	5	对于 由 单自由度液浮陀螺仪 构成 的 平台式惯性导航系统 , H 调制 陀螺 监控 技术 可以 同时 对 三个 导航 陀螺 分别 进行 监控 和 漂移 自 补偿 , 它 采用 北向 和 方位 H 调制 陀螺 监控 方案 , 在 惯性 平台 台体 上 加装 北向 和 方位 监控 陀螺 , 对 北向 和 方位 导航 陀螺 进行 监控 , 实现 误差 自 补偿"
# relationidx, positionMatrix1, positionMatrix2, tokenMatrix = prt.process_one(line)
# relationidxs, positionMatrix1, positionMatrix2, tokenMatrix,sdpMatrix = prt.process_file("files/train.txt",True,'pkl/train2.pkl.gz')
kjml = KejModel()
# kjml.build_new_model()
# model = kjml.load_kej_model('model/kej_model.h5')
# print(model.summary())
# model = kjml.load_kej_model("model/kej_model.h5")
kjml.train_model(False,"model/kej_model3.h5",load = False)
# kjml.model_predict("pkl/test.pkl.gz")
# kjml.model_predict_one(model,relationidx, positionMatrix1, positionMatrix2, tokenMatrix)

