import math

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures


x = [0.024, 0.038, 0.04, 0.045, 0.047, 0.0578, 0.0629, 0.0629, 0.063, 0.064,
0.0678, 0.0691, 0.071, 0.0742, 0.0752, 0.077, 0.0779, 0.0781, 0.0787, 0.0789,
0.0791, 0.0862, 0.0867, 0.0877, 0.089, 0.0897, 0.096, 0.098, 0.099]


y = [11.7, 12.7, 15.5, 16.8, 16.7, 17.5, 18.5, 18.7, 18.8, 19.5, 20.8, 20.3,
23.3, 23.2, 23.7, 24.4, 28.9, 25.8, 29.5, 23.3, 22.5, 26.2, 29.7, 33.8, 35,
32, 40, 41, 43.8]

x_np = np.array(x).reshape((-1,1))
y_np = np.array(y)

model = LinearRegression().fit(x_np,y_np)
r_sq = model.score(x_np,y_np)

a0 = round(model.intercept_,2)
a1 = round(model.coef_[0],2)
y_pred = model.predict(x_np)

print('Коефіцієнт Пірсона (кореляції) для цих даних: {},'.format(round(pearsonr(x,y)[0],3)) + ' число близьке до одиниці, можна використати лінійну регресію')
print('Рівняння лінійної залежності y = {} + {}x'.format(a0,a1))

print('Коефіцієнт детермінації, лінійної регресії: {}'.format(r_sq))


# linear_correlation = plt.figure(1)
# plt.scatter(x,y)
# plt.plot(x,y_pred)
# plt.show()


x_ = PolynomialFeatures(degree=2,include_bias=False).fit_transform(x_np)

model2 = LinearRegression().fit(x_,y_np)

r_sq2 = model2.score(x_,y_np)

b0 = model2.intercept_
b1 = model2.coef_

log_x_data = np.log(np.array(x))
log_y_data = np.log(y_np)

model3 = LinearRegression().fit(log_x_data.reshape(-1,1),y_np)

print(model3.score(log_x_data.reshape(-1,1),y_np))
print(model3.intercept_,model3.coef_)

# curve_fit = np.polyfit(log_x_data,y_np,1)
# print(curve_fit)

# polynomial_correlation = plt.figure(2)
# y_pred2 = model2.predict(x_)
# plt.scatter(x,y)
# plt.plot(x,y_pred2)
# plt.show()




# def logarithmic(x,y):
#     sum_y = sum(y)
#     sum_lnx2 = sum([math.log(xs,math.e) for xs in x])
#     sum_lnx3 = sum([pow(math.log(xs,math.e),2) for xs in x])







