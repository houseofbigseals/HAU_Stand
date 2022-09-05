
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

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


def draw_rv2_pressure():
    # data
    volume = np.array([
        100, 160, 200, 225, 50, 100, 150, 200, 225, 50, 75, 100, 125, 150, 175
    ])  # water volume pumped into rv

    pressure = np.array([
        35, 60, 85, 110, 8, 25, 60, 85, 115, 8, 15, 30, 49, 60, 78
    ])  # pressure in rv by external analog manometer data

    def quad_func(t, a, b, c):
        return a*t*t + b*t + c

    print(len(volume))
    print(len(pressure))

    popt, r2 = approximation_with_r2(quad_func, volume, pressure)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    print(popt, r2)

    plt.plot(volume, pressure,  'ob', label="Данные манометра")
    yerr = np.ones(np.shape(pressure)[0])*5
    # plt.errorbar(volume, quad_func(volume, *popt), yerr=yerr, fmt='v',
    #              linestyle='', color='g', label='Погрешности измерения', capsize=5)
    plt.errorbar(volume, pressure, yerr=yerr, #fmt='v',
                               linestyle='', color='g', label='Погрешности измерения', capsize=5)

    line_label = f"Аппроксимация y = {a:.3}*x^2 + {b:.3}*x + {c:.3}, R2 = {r2:.4}"

    pseudo_vol = np.arange(50, 230, 5)

    plt.plot(pseudo_vol, quad_func(pseudo_vol, *popt), '-r', label=line_label)

    plt.ylabel('Давление, кПа')
    plt.xlabel('Закачанный в РВ объем, мл')
    plt.title("Давление в гидросистеме РВ2 при наполнении водой")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def draw_s2_plot_180ml():
    # SENSOR 2 !

    # pumping in

    v_in = np.arange(0, 190, 10)  # pumped in water volume in ml
    press_in = np.array([
        2.35, 2.40, 2.43, 2.47, 2.52, 2.54, 2.58, 2.63, 2.70, 2.74, 2.81, 2.85,
        2.91, 2.98, 3.06, 3.13, 3.21, 3.31, 3.40
    ])

    # pumping out

    v_out = np.arange(180, -10, -10)
    press_out = np.array(
        [3.37, 3.27, 3.19, 3.12, 3.06, 2.93, 2.86, 2.80, 2.74, 2.68, 2.63, 2.59, 2.54,
         2.50, 2.47, 2.42, 2.39, 2.36, 2.32]
    )


    # in
    plt.plot(v_in, press_in,  '-og', label="Заполнение")
    plt.plot(v_out, press_out, '-or', label="Опустошение")

    # line_label = f"Linear approximation y = {a:.3}*x + {b:.3}, R2 = {r2:.4}"

    # plt.plot(press_1, lin_func(np.array(press_1), *popt), '-vr', label=line_label)

    plt.xlabel('Объем воды в РВ, мл')
    plt.ylabel('Показания датчика давления, V')
    plt.title("Наполнение и опустошение резервуара №2 на 180 мл")
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # draw_s2_plot_180ml()
    draw_rv2_pressure()
