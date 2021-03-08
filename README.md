# 经典不平衡分类机器学习算法
## 数据层面
随机欠采样（Random Under-Sampling）   
随机过采样（Random Over-Sampling）   
基于聚类的过采样（Cluster-Based Over Sampling）   
合成少数类过采样（SMOTE）   

## 算法层面
Bagging   
Boosting


## 分类器
k-Nearest Neighbors （K近邻算法）
Support Vector Machines (SVM)（支持向量机）
Decision Trees（决策树）

# 项目环境
sklearn   
imblearn   

# 遇到的问题
Q：为什么要进行降采样？（不平衡分类会导致什么问题？）   
A：

Q: 降采样率设为多少合适？   
个人猜测：
不平衡率较小时：设为平衡采样率   
不平衡率较大时：先对少数类进行过采样，使不平衡率较小，然后再按平衡采样率进行采样。