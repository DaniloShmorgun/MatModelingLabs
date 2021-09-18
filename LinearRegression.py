import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


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


linear_correlation = plt.figure(1)
linear_correlation.scatter(x,y)
plt.plot(x,y_pred)
plt.show()
# mean_x = np.mean(x)
# mean_y = np.mean(y)

