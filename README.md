# Machine Learning in Action
- Python3 
- Machine Learning in Action

# 机器学习模型
## 监督学习
- 分类：线性分类器（如LR)、支持向量机（SVM）、朴素贝叶斯（NB）、K近邻（KNN）、决策树（DT）、集成模型（RF/GDBT等）
- 回归：线性回归、支持向量机（SVM）、K近邻（KNN）、回归树（DT）、集成模型（ExtraTrees/RF/GDBT）
## 无监督模型
- 数据聚类（K-means）/ 数据降维（PCA）等等.
##

模型名称|数学假设|模型优缺点|评测指标及其计算方法
:-:|:-:|:-:|:-:
LR|假设特征与分类结果存在线性关系使用sigmoid函数映射到0-1|与随机梯度上升算法相比，预测精度准确，但是耗费时间长|准确性（Accuracy）<br>召回率（Recall）<br>精确率（Precision）以及<br>F1
NB|各个维度上的特征被分类的条件概率之间是相互独立的、贝叶斯公式|广泛用于文本分类<br>优点：速度快，参数估计的个数锐减<br>缺点：在特征关联性较强的任务性能差|同上
集成模型|训练多个模型<br>RF---bagging<br>GDBT----boosting<br>模型融合相关内容|优点：性能高、稳定性强、广泛应用于工业界<br>缺点：训练时间长，调参是体力活<br>xgb、lightGBM是比较快的|同上
回归相关的模型|SVM有三种核函数<br>（linear/poly/rbf）| |R^2/MAE/MSE/RMSE
