"""
Charles Habermehl
2018-04-13
CS 365
Homework 4
"""

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

"""
Part A

Input W = w     Initial Gate = g1       Waste w = t1 
Input X = x     Intermediate Gate = g2  Waste x = t2
Input Y = y     Intermediate Gate = g3  Unreactive Strand = t3
Output Z = z    Intermediate Gate = g4  Unreactive Gate = t4
Fuel = f

w + g1 -> t1 + g2   t1 + g2 -> w + g1
x + g2 -> t2 + g3   t2 + g3 -> x + g2
y + g3 -> z + g4    z + g4 -> y + g3
f + g4 -> t3 + t4

d[w]/dt = -k[w][g1] + k[t1][g2]
d[x]/dt = -k[x][g2] + k[t2][g3]
d[y]/dt = -k[y][g3] + k[z][g4]
d[z]/dt = k[y][g3] - k[z][g4]
d[f]/dt = -k[f][g4]
d[g1]/dt = -k[w][g1] + k[t1][g2]
d[g2]/dt = k[w][g1] - k[t1][g2] - k[x][g2] + k[t2][g3]
d[g3]/dt = k[x][g2] - k[t2][g3] - k[y][g3] + k[z][g4]
d[g4]/dt = k[y][g3] - k[z][g4] - k[f][g4]
d[t1]/dt = k[w][g1] - k[t1][g2]
d[t2]/dt = k[x][g2] - k[t2][g3]
d[t3]/dt = k[f][g4]
d[t4]/dt = k[f][g4]

Part B
      0    1    2    3    4    5     6     7     8     9     10    11    12
y = {[w], [x], [y], [z], [f], [g1], [g2], [g3], [g4], [t1], [t2], [t3], [t4]}

dy/dt = {   
            -k*y[0]*y[5] + k*y[9]*y[6]
            -k*y[1]*y[6] + k*y[10]*y[7]
            -k*y[2]*y[7] + k*y[3]*y[8]
            k*y[2]*y[7] - k*y[3]*y[8]
            -k*y[4]*y[8]
            -k*y[0]*y[5] + k*y[9]*y[6]
            k*y[0]*y[5] - k*y[9]*y[6] - k*y[1]*y[6] + k*y[10]*y[7]
            k*y[1]*y[6] - k*y[10]*y[7] - k*y[2]*y[7] + k*y[3]*y[8]
            k*y[2]*y[7] - k*y[3]*y[8] - k*y[4]*y[8]
            k*y[0]*y[5] - k*y[9]*y[6]
            k*y[1]*y[6] - k*y[10]*y[7]
            k*y[4]*y[8]
            k*y[4]*y[8]
        }        
"""


# single vector ODE function
def f(y, t, k):
    return np.array([-k * y[0] * y[5] + k * y[9] * y[6],
                     -k * y[1] * y[6] + k * y[10] * y[7],
                     -k * y[2] * y[7] + k * y[3] * y[8],
                     k * y[2] * y[7] - k * y[3] * y[8],
                     -k * y[4] * y[8],
                     -k * y[0] * y[5] + k * y[9] * y[6],
                     k * y[0] * y[5] - k * y[9] * y[6] - k * y[1] * y[6] + k * y[10] * y[7],
                     k * y[1] * y[6] - k * y[10] * y[7] - k * y[2] * y[7] + k * y[3] * y[8],
                     k * y[2] * y[7] - k * y[3] * y[8] - k * y[4] * y[8],
                     k * y[0] * y[5] - k * y[9] * y[6],
                     k * y[1] * y[6] - k * y[10] * y[7],
                     k * y[4] * y[8],
                     k * y[4] * y[8]])


def main():
    # for naming my plots
    file_name_template = r'HW4_plot{0:02d}.pdf'

    # true/false initial inputs
    initial_inputs = [(0, 0, 0),
                      (0, 0, 1),
                      (0, 1, 0),
                      (1, 0, 0),
                      (1, 1, 0),
                      (0, 1, 1),
                      (1, 0, 1),
                      (1, 1, 1)]
    # plot coounter
    idx = 0

    # Part C
    for (w, x, y) in initial_inputs:
        # a bunch of initializations derived from the lecture notes
        signal_amt = 10
        gate_fuel = 100
        w_0, x_0, y_0, z_0 = w * signal_amt, x * signal_amt, y * signal_amt, 0
        g1_0, g2_0, g3_0, g4_0 = gate_fuel, 0, 0, 0
        f_0 = gate_fuel
        t1_0, t2_0, t3_0, t4_0 = 0, 0, 0, 0
        y0 = np.array([w_0, x_0, y_0, z_0, f_0, g1_0, g2_0, g3_0, g4_0, t1_0, t2_0, t3_0, t4_0])
        sim_length = 7200
        ts = np.linspace(0, sim_length, sim_length + 1)
        k = 5e-4  # 5e5 /M/s = 5e-4 /nM/s

        # integrate
        res = integrate.odeint(f, y0, ts, args=(k,))
        w_vals = res[:, 0]
        x_vals = res[:, 1]
        y_vals = res[:, 2]
        z_vals = res[:, 3]

        # plot
        plt.figure(idx)
        plt.plot(ts, w_vals, color='red', linewidth=2, label='[W]')
        plt.plot(ts, x_vals, color='green', linewidth=2, label='[X]')
        plt.plot(ts, y_vals, color='blue', linewidth=2, label='[Y]')
        plt.plot(ts, z_vals, color='black', linewidth=2, label='[Z]')
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration')
        plt.title('w={}, x={}, y={}'.format(w, x, y))
        plt.legend()
        plt.savefig(file_name_template.format(idx), bbox_inches='tight')
        plt.clf()
        idx += 1

    # Part D
    # many more concentrations
    conc = [10, 25, 50, 100, 150, 200]
    y_pos = np.arange(len(conc))
    z_percents = []
    zz_vals = []

    # calculate output values based on the new concentrations
    for i in range(len(conc)):
        for (ww, xx, yy) in initial_inputs:
            signal_value = conc[i]
            gate_fuel_value = 100
            ww_0, xx_0, yy_0, zz_0 = ww * signal_value, xx * signal_value, yy * signal_value, 0
            gg1_0, gg2_0, gg3_0, gg4_0 = gate_fuel_value, 0, 0, 0
            ff_0 = gate_fuel_value
            tt1_0, tt2_0, tt3_0, tt4_0 = 0, 0, 0, 0
            yy0 = np.array([ww_0, xx_0, yy_0, zz_0, ff_0, gg1_0, gg2_0, gg3_0, gg4_0, tt1_0, tt2_0, tt3_0, tt4_0])
            sim_length = 7200
            tts = np.linspace(0, sim_length, sim_length + 1)
            kk = 5e-4

            resres = integrate.odeint(f, yy0, tts, args=(kk,))
            zz_vals = resres[:, 3]

        # take last z value and use it to make a yield percentage
        z_percent = (zz_vals[-1] / conc[i]) * 100
        z_percents.append(z_percent)

    # bar chart the yield percentage
    plt.figure(idx)
    plt.bar(y_pos, z_percents, align='center', alpha=0.5)
    plt.xticks(y_pos, conc)
    plt.xlabel('Initial Concentration (nM)')
    plt.ylabel('Percentage Output (%)')
    plt.title('Z Percentage Output Chart')
    plt.savefig('percent_output.pdf')
    plt.clf()


"""
Part E
The greater the input concentration we have the more waste/leakage we will have which leads to lower output percentages.
"""

if __name__ == '__main__':
    main()
