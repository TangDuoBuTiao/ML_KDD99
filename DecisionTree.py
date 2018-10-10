import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split

'''
使用决策树的方法训练模型
最后输出的预测精度是Predict accuracy: 0.9999499840255102                                                  
'''

# 读取数据
data_dir = "./"
raw_data_filename = data_dir + "kddcup.data"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None)

'''
将非数值型的数据转换为数值型数据
0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
'''
print("Transforming data...")
raw_data[1], protocols = pd.factorize(raw_data[1])  # factorize()方法将非数值型数据映射成数字，返回值是一个元组
raw_data[2], services = pd.factorize(raw_data[2])
raw_data[3], flags = pd.factorize(raw_data[3])
raw_data[41], attacks = pd.factorize(raw_data[41])

# 对原始数据进行切片，分离出特征和标签，第1~41列是特征，第42列是标签
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # pandas中的iloc切片是完全基于位置的索引
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]

# 将多维的标签转为一维的数组
labels = labels.values.ravel()

# 将数据分为训练集和测试集,并打印维数
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2)
print("X_train,y_train:", X_train.shape, y_train.shape)
print("X_test,y_test:", X_test.shape, y_test.shape)

# 模型的参数全部为默认值

print("Training model...")
clf = DecisionTreeClassifier(criterion='gini')
trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))

# predicting
print("Predicting...")
y_pred = clf.predict(X_test)

print("Computing performance metrics...")
results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)

# 根据混淆矩阵求预测精度

list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy: ", np.mean(list_diag) / np.mean(list_raw_sum))
