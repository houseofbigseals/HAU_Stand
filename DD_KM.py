
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# data
# stable measured data - y

level = [50, 40, 30, 20, 10, 0]  # cm
pressure = [-5, -4, -3, -2, -1, 0]  # kPa

# experimental data - to calibrate - x
# sensor 4, V

press_1 = [0.91, 1.67, 2.39, 3.21, 3.81, 4.53]
press_2 = [0.96, 1.60, 2.29, 3.14, 3.82, 4.60]
press_3 = [0.90, 1.62, 2.35, 3.13, 3.82, 4.61]
press_4 = [0.89, 1.66, 2.40, 3.14, 3.87, 4.56]
press_5 = [0.88, 1.65, 2.39, 3.09, 3.82, 4.56]

full_s4_data = np.array([*press_1, *press_2, *press_3, *press_4, *press_5])
full_y_data = np.array([*pressure, *pressure, *pressure, *pressure, *pressure])

# sensor 3, V

press_6 = [1.02, 1.88, 2.68, 3.35, 4.10, 4.79]
press_7 = [1.03, 1.86, 2.57, 3.28, 4.11, 4.75]
press_8 = [0.95, 1.70, 2.60, 3.30, 4.07, 4.80]
press_9 = [1.07, 1.80, 2.61, 3.38, 4.13, 4.81]
press_10 = [1.01, 1.72, 2.58, 3.27, 4.03, 4.79]

full_s3_data = np.array([*press_6, *press_7, *press_8, *press_9, *press_10])


def approximation_with_r2(func, x, y):
    popt, pcov = curve_fit(func, x, y)
    print("popt using scipy: {}".format(popt))
    print("pcov using scipy: {}".format(pcov))
    # perr = np.sqrt(np.diag(pcov))
    # print("perr using scipy: {}".format(perr))

    # to compute R2
    # https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit

    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print("r_squared using custom code: {}".format(r_squared))
    return popt, r_squared


def draw_s4_plot():

    def lin_func(t, a, b):
        return a*t + b

    print(len(full_y_data))
    print(len(full_s4_data))
    popt, r2 = approximation_with_r2(lin_func, full_s4_data, full_y_data)
    a = popt[0]
    b = popt[1]
    print(popt, r2)

    plt.plot(press_1, pressure,   'ob')
    plt.plot(press_2, pressure,  'ob')
    plt.plot(press_3, pressure,  'ob')
    plt.plot(press_4, pressure,  'ob')
    plt.plot(press_5, pressure,  'ob', label="Data")

    line_label = f"Linear approximation y = {a:.3}*x + {b:.3}, R2 = {r2:.4}"

    plt.plot(press_1, lin_func(np.array(press_1), *popt), '-vr', label=line_label)

    plt.ylabel('Разрежение, кПа')
    plt.xlabel('Показания датчика давления, V')
    plt.title("Калибровка датчика давления №4")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def draw_s3_plot():

    def lin_func(t, a, b):
        return a*t + b

    print(len(full_y_data))
    print(len(full_s3_data))
    popt, r2 = approximation_with_r2(lin_func, full_s3_data, full_y_data)
    a = popt[0]
    b = popt[1]
    print(popt, r2)

    plt.plot(press_6, pressure,   'ob')
    plt.plot(press_7, pressure,  'ob')
    plt.plot(press_8, pressure,  'ob')
    plt.plot(press_9, pressure,  'ob')
    plt.plot(press_10, pressure,  'ob', label="Data")

    line_label = f"Linear approximation y = {a:.3}*x + {b:.3}, R2 = {r2:.4}"

    plt.plot(press_10 , lin_func(np.array(press_10), *popt), '-vr', label=line_label)

    plt.ylabel('Разрежение, кПа')
    plt.xlabel('Показания датчика давления, V')
    plt.title("Калибровка датчика давления №3")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()





if __name__ == "__main__":
    draw_s4_plot()
    draw_s3_plot()