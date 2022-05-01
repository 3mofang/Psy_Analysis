import pymysql as pm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import math

np.set_printoptions(threshold=np.inf)

n = 20  # 心理量表题目数量
M = 258  # 量表统计数据量
PY = 0.5  # 先验概率
P_Y = 1 - PY
Psy_flag = 17 / 20  # 社会适应能力表 正常异常区分值


# X(in):测试者选择的每题的选项，定义域{-2,0,2}
# Y(out):SCL-90 的“人际关系敏感”倾向题，9道总共均值在

def db():
    "从数据库中提取数据 以一维数据流输入，转换为M*(n+1)的二维列表，进行输出"
    db = pm.connect(host="localhost", user="root", password="123456", database="psychomeasure_scale")

    cursor = db.cursor()

    sql = "SELECT `scl-90_tag`,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20 FROM psytest_using"

    try:
        cursor.execute(sql)
        pnlist = []
        results = cursor.fetchall()
        for row in results:
            for column in range(n + 1):
                pnlist.append(row[column])

        # print('列表总长度：',len(pnlist))

    except:
        print("Error: unable to fetch data")

    db.commit()
    cursor.close()
    db.close()

    return pnlist

def db_to_array(pnlist):
    inlist = np.array(pnlist)  # 建立numpy.array对象

    L = inlist.reshape((M, n + 1))  # List,一维数组二维化，将一维列表转换为258*21(即m,n+1)的列表
    # print(list)

    return L

def train(L):
    "输入训练集，使用朴素贝叶斯分类器对权重进行训练，输出两种权重计算方式计算出的权重"
    m = len(L)  # 训练集行数

    N = np.empty((n, 4, 3), dtype=int)
    # Number,N[20][4][3]是两种类型（SCL-90划分）的人分别在每题上选择选项的数量统计数组
    # 20道题，每题3个选项，2种类型，共20（道题）*4（3(分别选择3种选项的每项的人数）+1（所有选择三种选项的某种人数量））*3（2(种人)+1(2种人选择该选项的合计)）=240（个数据）
    N.fill(0)  # 三维数组初始化
    # print(N)

    for j in range(n):
        for i in range(M):
            if L[i][0] == 0 and L[i][j + 1] == -2:
                N[j][0][0] = N[j][0][0] + 1
            elif L[i][0] == 0 and L[i][j + 1] == 0:
                N[j][1][0] = N[j][1][0] + 1
            elif L[i][0] == 0 and L[i][j + 1] == 2:
                N[j][2][0] = N[j][2][0] + 1
            elif L[i][0] == 1 and L[i][j + 1] == -2:
                N[j][0][1] = N[j][0][1] + 1
            elif L[i][0] == 1 and L[i][j + 1] == 0:
                N[j][1][1] = N[j][1][1] + 1
            elif L[i][0] == 1 and L[i][j + 1] == 2:
                N[j][2][1] = N[j][2][1] + 1
    # print(N)

    for j in range(n):
        N[j][0][2] = N[j][0][0] + N[j][0][1]
        N[j][1][2] = N[j][1][0] + N[j][1][1]
        N[j][2][2] = N[j][2][0] + N[j][2][1]
        N[j][3][0] = N[j][0][0] + N[j][1][0] + N[j][2][0]
        N[j][3][1] = N[j][0][1] + N[j][1][1] + N[j][2][1]
        N[j][3][2] = M
    # print(N)

    P = np.empty((n, 3, 3), dtype=float)
    # Probability,P[20][3][3]是条件概率(似然函数)数组
    # 20道题，每题3个选项，2种类型，共20（道题）*3（在种类为0时选择-2的概率，以此类推*3）*3（2种类型的人+由全概率公式推算出的选择每种选项的概率）=180（个数据）

    P.fill(0.0)  # 三维数组初始化
    # print(P)

    # 计算似然：
    for j in range(n):
        P[j][0][0] = N[j][0][0] / N[j][3][0]
        P[j][1][0] = N[j][1][0] / N[j][3][0]
        P[j][2][0] = N[j][2][0] / N[j][3][0]
        P[j][0][1] = N[j][0][1] / N[j][3][1]
        P[j][1][1] = N[j][1][1] / N[j][3][1]
        P[j][2][1] = N[j][2][1] / N[j][3][1]

        P[j][0][2] = P[j][0][0] * PY + P[j][0][1] * P_Y
        P[j][1][2] = P[j][1][0] * PY + P[j][1][1] * P_Y
        P[j][2][2] = P[j][2][0] * PY + P[j][2][1] * P_Y
    # print(P)

    PO = np.empty((n, 3, 2), dtype=float)
    # POsteriori,后验概率，O[20][3][2]是进行贝叶斯计算后输出的后验概率。
    # 此数据可以用来说明，当X取值分别为-2，0，2时，在多大程度上可以说明，Y是0或1
    # 20道题，每题3个选项，2种类型，共20（道题）*3（X分别是-2，0，2时，对Y的后验概率P(Y=0|X)）*2（2种Y）

    PO.fill(0.0)  # 三维数组初始化
    # print(PO)

    for j in range(n):
        PO[j][0][0] = P[j][0][0] / P[j][0][2] * PY
        PO[j][1][0] = P[j][1][0] / P[j][1][2] * PY
        PO[j][2][0] = P[j][2][0] / P[j][2][2] * PY
        PO[j][0][1] = P[j][0][1] / P[j][0][2] * P_Y
        PO[j][1][1] = P[j][1][1] / P[j][1][2] * P_Y
        PO[j][2][1] = P[j][2][1] / P[j][2][2] * P_Y

    S = np.empty((n, 2), dtype=float)
    # Support,支持度，取Y=0时3种后验概率之和作为本题所有选项对其支持度。
    # 20道题，每题2种类型，共20*2=40（种数据）

    S.fill(0.0)  # 三维数组初始化
    # print(PO)

    for j in range(n):
        S[j][0] = PO[j][0][0] + PO[j][1][0] + PO[j][2][0]
        S[j][1] = PO[j][0][1] + PO[j][1][1] + PO[j][2][1]

    # for j in range(20):
    #     print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ".format(PO[j][0][0], PO[j][1][0], PO[j][2][0], PO[j][0][1],
    #                                                               PO[j][1][1], PO[j][2][1]))

    D1 = np.empty((n), dtype=float)
    # discrimination，使用两种支持度之差作为第一种区分度
    # 共20道题，每题对应一种区分度。

    D1.fill(0.0)

    D1n = 0  # D1区分度之和，用于计算最终权重值。
    for j in range(n):
        D1[j] = abs(S[j][0] - S[j][1])  # X对Y正常的支持度（Y=0）-X对Y异常的支持度/置信度（Y=1） 由于需要的是“区分度”，因此取绝对值，另外正负可以判断该题目倾向方向
        D1n = D1n + D1[j]

    W1 = np.empty((n), dtype=float)
    # Weight,权重，对应每道题的权重
    # 共20道题，每题对应一种区分度。

    W1.fill(0.0)
    for j in range(n):
        W1[j] = D1[j] / D1n

    # print(W1)

    D2 = np.empty((n), dtype=float)
    # discrimination,使用后验几率，即后验概率的比值来描述支持度/置信度，比值与1之间的大小比较可以说明倾向方向

    D2.fill(0.0)
    # print(D2)

    k = 0  # 夸张度
    D2n = 0  # D2区分度之和，用于计算最终权值
    for j in range(n):
        # D2[j]=(S[j][0] / S[j][1])
        D2[j] = abs(math.log((S[j][0] / S[j][1]) + k, math.e))
        # D2[j] = math.log(S[j][0] + k, math.e) / math.log(S[j][1] + k, math.e)
        # D2 = math.exp(S)
        # D2n = D2[j] + D2n
        D2n = math.exp(D2[j]) + D2n
        # print(D2n)
        # print("{:.3f}".format(D2[j]))

    W2 = np.empty((n), dtype=float)
    # Weight,权重，对应每道题的权重
    # 共20道题，每题对应一种区分度。

    W2.fill(0.0)
    for j in range(n):
        W2[j] = math.exp(D2[j]) / D2n  # 使用softmax方法对后验几率进行归一化

    print(W1)
    print(W2)
    X = range(0, n)
    # plt.plot(X, W1)
    plt.plot(X, W2)
    plt.show()

    return (W1, W2)


def test(test_list, W):
    "输入测试集、权重，输出以Psy_flag为分界线的社会适应能力结果分类标签"
    L = test_list[:, 1:21]  # 将SCL_()_tag切掉
    m = len(L)

    flag = np.empty((m), dtype=int)

    flag.fill(0)

    for i in range(m):
        # print("L{}:{}".format(i,L[i]))
        # print(W)
        sum = np.dot(L[i], W)
        print("{:.4f}".format(sum * 20))
        if sum > Psy_flag:
            flag[i] = 0
        else:
            flag[i] = 1
    return flag


def model_1():
    "不加性别分类的原始权重生成模式"
    pnlist = db()  # 从数据库中提取数据 以一维列表形式输出提取的数据流 再将其转换为M*（n+1）二维列表
    # print(db.__doc__)

    L=db_to_array(pnlist)
    # print(L)

    W1, W2 = train(L)  # 输入258*21二维数组，输出计算得出的权值

    flag = test(L, W2)

    print(flag)


def model_2():
    "加入性别分类的权重生成模式"


def main():
    model_1()


if __name__ == '__main__':
    main()
