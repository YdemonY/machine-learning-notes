from sklearn.linear_model import LinearRegression
import numpy as np

# 简单一元线性回归示例
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print("系数（w）：", model.coef_)
print("截距（b）：", model.intercept_)
print("预测 x=6：", model.predict([[6]]))
