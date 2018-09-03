# 构建单层分类器
# 单层分类器是基于最小加权分类错误率的树桩
# 伪代码
# 将最小错误率minError设为+∞
# 对数据集中的每个特征(第一层特征)：
#      对每个步长(第二层特征)：
#          对每个不等号(第三层特征)：
#                建立一颗单层决策树并利用加权数据集对它进行测试
#                如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
# 返回最佳单层决策树

from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


# 自适应加载数据
def loadDataSet(fileName):  # general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t'))  # get number of fields
    # 创建数据集矩阵，标签向量
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    # 遍历文本每一行
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        # 数据矩阵
        dataMat.append(lineArr)
        # 标签向量
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


#   单层决策树生成函数
#   单层决策树的阈值过滤函数
#   通过阈值比较对数据进行分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # just classify the data
    retArray = ones((shape(dataMatrix)[0], 1))      # 首先将返回数组的全部元素设置为1
    if threshIneq == 'lt':                          # 不满足不等式要求的元素设置为-1，可以基于数据集中的任意元素进行比较
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


#   遍历stumpClassify()函数所有的可能输入值，并找到数据集上最佳的单层决策树
def buildStump(dataArr, classLabels, D):
    # 将数据集和标签列表转为矩阵形式
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # 步长或区间总数 最优决策树信息 最优单层决策树预测结果
    numSteps = 10.0;
    bestStump = {};                 # 存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))
    # 最小错误率初始化为+∞
    minError = inf  # init error sum, to +infinity
    for i in range(n):  # loop over all dimensions (在数据集的所有特征上遍历)
        # 找出列中特征值的最小值和最大值
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max();
        # 求取步长大小或者区间间隔
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历各个步长区间
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            # 两种阈值过滤模式
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                # 阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                threshVal = (rangeMin + float(j) * stepSize)
                # 选定阈值后，调用阈值过滤函数分类预测
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # call stump classify with i, j, lessThan
                # 初始化错误向量
                errArr = mat(ones((m, 1)))              # 如果predictedVals中的值不等于labelMat中真正类别标签值，则errArr的相应位置为1
                # 将错误向量中分类正确项置0
                errArr[predictedVals == labelMat] = 0
                # 计算“加权”的错误率
                weightedError = D.T * errArr  # calc total error multiplied by D
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 将当前的错误率与已有的最小错误率进行对比，如果当前的值较小，那么就在词典bestStump中保存该单层决策树
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    return bestStump, minError, bestClasEst


#   基于单层决策树的AdaBoost训练过程
#   dataArr:数据集 classLabels:类别标签 numIt:迭代次数
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    # 弱分类器相关信息列表
    weakClassArr = []       # 弱分类器数组
    # 获取数据集行数
    m = shape(dataArr)[0]
    # 初始化权重向量的每一项值相等
    D = mat(ones((m,1))/m)   # init D to all equal   (D是概率分布向量)初始化权重
    # 累计估计值向量
    aggClassEst = mat(zeros((m,1)))     # aggClassEst记录每个数据点的类别估计累计值
    # 循环迭代次数
    for i in range(numIt):
        # 根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)  # build Stump 建立单层决策树
        # 打印权重向量
        print("D:",D.T)
        # 求单层决策树的系数alpha
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))    # calc alpha, throw in max(error,eps) to account for error=0 计算当前分类器的权重(避免发生除零溢出)
        # 存储决策树的系数alpha到字典
        bestStump['alpha'] = alpha
        # 将该决策树存入列表
        weakClassArr.append(bestStump)                  # store Stump Params in Array 存储当前分类器
        # 打印决策树的预测结果
        print("classEst: ",classEst.T)
        # 预测正确为exp(-alpha),预测错误为exp(alpha)
        # 即增大分类错误样本的权重，减少分类正确的数据点权重
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)  # exponent for D calc, getting messy
        # 更新权重向量
        D = multiply(D,exp(expon))                              # Calc New D for next iteration 更新样本权重值
        D = D/D.sum()
        # 累加当前单层决策树的加权预测值
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst       # 更新当前组合分类器的决策
        print( "aggClassEst: ",aggClassEst.T)
        # 求出分类错的样本个数
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        # 计算错误率
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break      # 总错误率为0，则由break语句中止for循环
    return weakClassArr,aggClassEst


#   AdaBoost分类函数
#   利用训练得到的弱分类器组合成强分类器进行分类
#   datToClass:待分类样例    classifierArr:多个弱分类器组成的数组
def adaClassify(datToClass,classifierArr):
    # 构建数据向量或矩阵
    dataMatrix = mat(datToClass)     # do stuff similar to last aggClassEst in adaBoostTrainDS(将datToClass转换成一个NumPy矩阵)
    # 获取矩阵行数
    m = shape(dataMatrix)[0]         # 待分类样例的个数m
    # 初始化最终分类器
    aggClassEst = mat(zeros((m,1)))  # 0列向量
    # 遍历classifierArr中的所有弱分类器，并基于stumpClassify()对每个分类器得到一个类别的估计值
    for i in range(len(classifierArr)):
        # 每一个弱分类器对测试数据进行预测分类
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])  # call stump classify
        # 对各个分类器的预测结果进行加权累加
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    # 通过sign函数根据结果大于或小于0预测出+1或-1
    return sign(aggClassEst)


# ROC曲线的绘制及AUC计算函数
# preStrengths：行向量矩阵，分类器的预测强度
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    # 构建浮点数二元组，初始化为(1,0,1.0)，该元组保留的是绘制光标的位置
    cur = (1.0,1.0) #cursor
    # ySum用于计算AUC的值
    ySum = 0.0 #variable to calculate AUC
    # 通过数组过滤方式计算正例的数目，并将该值赋给numPosClas
    numPosClas = sum(array(classLabels)==1.0)
    # 在x轴和y轴的0.0到1.0区间上绘点，y轴上的步长是1.0/numPosClas
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    # 获取排好序的索引，构建画笔
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 在所有排序值上循环
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        # 每得到一个标签为1.0的类，则沿着y轴方向下降一个步长，即不断降低真阳率
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        # 同理假阳率
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)
