from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import jieba

def datasets():
    iris = load_iris()
    # print("数据集", iris)
    # print("数据集", iris["DESCR"])

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值", x_train, x_train.shape)
    return None


def dict_demo():
    data = [{"city": "北京", "tempeature": 100}, {"city": "上海", "tempeature": 60}, {"city": "深圳", "tempeature": 80}]
    # 实例化一个转换器
    # sparse稀疏矩阵 Flase不为稀疏
    transfer = DictVectorizer(sparse=False)
    # 调用fit_transform ===>转为one-hot编码
    data_new = transfer.fit_transform(data)
    print(transfer.get_feature_names_out())
    print(data_new)


def text_demo():
    # 统计每个样本词出现的次数
    data = ["The swift breeze whispers through the trees", "carrying the scent of blooming flowers and the promise of a"
                                                           "new day."]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print(data_new.toarray())


def chinesetext_demo():
    # 统计每个样本词出现的次数
    data = ["从而顺利存储", "访问相应信息", "以便为您提供个性化和高品质的内容和产品或广告等服务"]
    text_list = []

    for text in data:
        text_list.append(cut_word(text))
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(text_list)
    print(transfer.get_feature_names_out())
    print(data_new.toarray())


def tdidf_demo():
    # 统计每个样本词出现的次数
    data = ["从而顺利存储", "访问相应信息", "以便为您提供个性化和高品质的内容和产品或广告等服务"]
    text_list = []

    for text in data:
        text_list.append(cut_word(text))
    transfer = TfidfVectorizer()
    data_new = transfer.fit_transform(text_list)
    print(transfer.get_feature_names_out())
    print(data_new.toarray())


def cut_word(text) -> str:
    text = " ".join(list(jieba.cut(text)))

    return text


def minmax_demo():
    # 1.获取数据
    data = pd.read_csv("data.txt")
    print("data:", data)
    data = data.iloc[:, :3]
    transfer = MinMaxScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


def standard_demo():
    # 1.获取数据
    data = pd.read_csv("data.txt")
    data = data.iloc[:, :3]
    transfer = StandardScaler()
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


def variance_demo():
    data = load_iris()
    # 过滤低方差特征
    transfmer = VarianceThreshold()
    new_data = transfmer.fit_transform(data.data)
    print(new_data)
    # 求相关系数
    r = pearsonr(data["变量1"], data["变量2"])


def PCAdemo():
    data = [[2, 8, 4, 5], [6, 3, 0, 8], 95, 4, 9, 1]
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(data)
    print(data_new)
    return None


if __name__ == '__main__':
    # datasets()
    # dict_demo()
    # text_demo()
    # chinesetext_demo()
    # tdidf_demo()
    # variance_demo()
    PCAdemo()
