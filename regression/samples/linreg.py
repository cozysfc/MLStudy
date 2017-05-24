# coding:utf-8


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_regressor import LinearRegressor


# Parameters
input_n = 2
epoch = 10000
alpha = 0.001

# Sample Dataset
train_x = [[1, 3], [2, 4], [2, 12]]
train_t = [2, 3, 7]

# Model
model = LinearRegressor(input_n=input_n, epoch=epoch, alpha=alpha)
model.build()

# Train
model.fit(train_x, train_t)

# Predict
pre = model.pre()
print pre([2, 3])
