from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    """
    Function returns the slope and y intercept for the best fit line of a set of data points
    Formula: m = Mean of x * Mean of y - Mean of xy /
                (Mean of x)^2 - Mean of (x^2)
            b = Mean of y - slope * Mean of x
    :param xs: x data points
    :param ys: y data points
    :return: the slope as m and the y-intercept as b
    """
    m = ((((mean(xs) * mean(ys))) - mean(xs * ys)) /
         ((mean(xs)*mean(xs)) - (mean(xs*xs))))
    b = mean(ys) - (m*mean(xs))
    return m, b


m, b = best_fit_slope_and_intercept(xs, ys)
# calculate all the y values
regression_line = [(m*x) + b for x in xs]

predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()
