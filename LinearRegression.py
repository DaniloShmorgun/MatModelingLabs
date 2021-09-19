import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
log_x_data = np.log(np.array(x))
log_y_data = np.log(y_np)

print('Коефіцієнт Пірсона (кореляції) для цих даних: {},'.format(round(pearsonr(x,y)[0],3)) + ' число близьке до одиниці,тоді можна використати лінійну регресію')

model_linear = LinearRegression().fit(x_np,y_np)
r2_linear = model_linear.score(x_np,y_np)
a0 = round(model_linear.intercept_,2)
a1 = round(model_linear.coef_[0],2)
y_pred_linear = model_linear.predict(x_np)
print('Рівняння лінійної залежності y = {} + {}x'.format(a0,a1))
print('Коефіцієнт детермінації, лінійної регресії: {}'.format(r2_linear))

x_ = PolynomialFeatures(degree=2,include_bias=False).fit_transform(x_np)
model_polynomial = LinearRegression().fit(x_,y_np)
r2_polynomial = model_polynomial.score(x_,y_np)
b0 = model_polynomial.intercept_
b1 = model_polynomial.coef_
y_pred_polynomial = model_polynomial.predict(x_)
print('Рівняння поліноміальної залежності y = {} * x^2 + {} * x + {}'.format(b1[0], b1[1],b0))
print('Коефіцієнт детермінації, поліноміальної регресії: {}'.format(r2_polynomial))

model_logarithm = LinearRegression().fit(log_x_data.reshape(-1,1),y_np)
c0 = model_logarithm.intercept_
c1 = model_logarithm.coef_[0]
logarithm_func = c0 + c1 * np.log(x_np)
r2_logarithmic = r2_score(y_np,logarithm_func)
print('Рівняння логарифмічної залежності y = {} * log(x) + {}'.format(c0,c1))
print('Коефіцієнт детермінації, логарифмічної регресії: {}'.format(r2_logarithmic))

model_exponent = LinearRegression().fit(x_np,log_y_data)
d0 = model_exponent.intercept_
d1 = model_exponent.coef_[0]
exponent_func = np.exp(d0 + x_np * d1)
r2_exponential = r2_score(y_np,exponent_func)
print('Рівняння лінійної залежності y = exp({} + x * {})'.format(d0,d1))
print('Коефіцієнт детермінації, експоненсіальної регресії: {}'.format(r2_exponential))

fig = plt.figure()
plt.xlabel("Залізо у питній воді")
plt.ylabel("Залізо у волоссі")
plt.title("Регресії")
plt.scatter(x,y)
plt.plot(x_np,y_pred_linear)
plt.annotate()
plt.plot(x_np,y_pred_polynomial)
plt.plot(x_np,logarithm_func)
plt.plot(x_np,exponent_func)
plt.show()

# print(model_exponent.score(x_np,log_y_data))
# print(model_exponent.intercept_,model_exponent.coef_)

# print(model_logarithm.score(log_x_data.reshape(-1,1),y_np))
# print(model_logarithm.intercept_,model_logarithm.coef_)



