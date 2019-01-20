"""
Charles Habermehl
2018-05-02
CS 365
Final Project
"""

import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

"""
INDIVIDUAL SPECIES:
    input 1 = x
    input 2 = y
    input 3 = z
    sum gate = s
    input:gate1,2,3 = ig1, ig2, ig3
    output1,2 = w1, w2
    threshhold = th
    waste1,2,3 = t1, t2, t3
    signal gate = sg
    signal:out = sgout
    fuel = f
    gate:fuel = gf
    fuel:out = fo
    reporter = r
    end output = eo

INDIVIDUAL REACTIONS:
    x + s -> ig1 + w1   ig1 + w1 -> x + s
    y + s -> ig2 + w1   ig2 + w1 -> y + s
    z + s -> ig3 + w1   ig3 + w1 -> z + s
    w1 + th -kfast-> t1 + t2
    w1 + sg -> sgout + w2   sgout + w2 -> w1 + sg
    f + sgout -> gf + w1    gf + w1 -> f + sgout
    w2 + r -> eo + t3

SPECIES EQUATIONS:
    d[x]/dt = -k[x][s] + k[ig1][w1]
    d[y]/dt = -k[y][s] + k[ig2][w1]
    d[z]/dt = -k[z][s] + k[ig3][w1]
    d[s]/dt = -k[x][s] + k[ig1][w1] - k[y][s] + k[ig2][w1] - k[z][s] + k[ig3][w1]
    d[ig1]/dt = k[x][s] - k[ig1][w1]
    d[ig2]/dt = k[y][s] - k[ig2][w1]
    d[ig3]/dt = k[z][s] - k[ig3][w1]
    d[w1]/dt = k[x][s] - k[ig1][w1] + k[y][s] - k[ig2][w1] + k[z][s] - k[ig3][w1] - kfast[w1][th] - k[w1][sg] + k[sgout][w2] + k[f][sgout] - k[gf][w1]
    d[w2]/dt = k[w1][sg] - k[sgout][w2]
    d[th]/dt = -kfast[w1][th]
    d[t1]/dt = kfast[w1][th]
    d[t2]/dt = kfast[w1][th]
    d[t3]/dt = k[w2][r]
    d[sg]/dt = -k[w1][sg] + k[sgout][w2]
    d[sgout]/dt = k[w1][sg] - k[sgout][w2] - k[f][sgout] + k[gf][w1]
    d[f]/dt = -k[f][sgout] + k[gf][w1]
    d[gf]/dt = k[f][sgout] - k[gf][w1]
    d[r]/dt = -k[w2][r]
    d[eo]/dt = k[w2][r]

ARRAY THAT DEFINES THE SPECIES:
          0    1    2     3    4      5      6      7     8     9    10    11    12    13     14      15    16   17   18
    y = {[x], [y], [z], [s], [ig1], [ig2], [ig3], [w1], [w2], [th], [t1], [t2], [t3], [sg], [sgout], [f], [gf], [r], [eo]}

ODES BASED ON THE SPECIES ARRAY AND EQUATIONS:
    dy/dt = {
                -k*y[0]*y[3] + k*y[4]*y[7]
                -k*y[1]*y[3] + k*y[5]*y[7]
                -k*y[2]*y[3] + k*y[6]*y[7]
                -k*y[0]*y[3] + k*y[4]*y[7] - k*y[1]*y[3] + k*y[5]*y[7] - k*y[2]*y[3] + k*y[6]*y[7]
                k*y[0]*y[3] - k*y[4]*y[7]
                k*y[1]*y[3] - k*y[5]*y[7]
                k*y[2]*y[3] - k*y[6]*y[7]
                k*y[0]*y[3] - k*y[4]*y[7] + k*y[1]*y[3] - k*y[5]*y[7] + k*y[2]*y[3] - k*y[6]*y[7] - kf*y[7]*y[9] - k*y[7]*y[13] + k*y[14]*y[8] + k*y[15]*y[14] - k*y[16]*y[7]
                k*y[7]*y[13] - k*y[14]*y[8]
                -kf*y[7]*y[9]
                kf*y[7]*y[9]
                kf*y[7]*y[9]
                k*y[8]*y[17]
                -k*y[7]*y[13] + k*y[14]*y[8]
                k*y[7]*y[13] - k*y[14]*y[8] - k*y[15]*y[14] + k*y[16]*y[7]
                -k*y[15]*y[14] + k*y[16]*y[7]
                k*y[15]*y[14] - k*y[16]*y[7]
                -k*y[8]*y[17]
                k*y[8]*y[17]
            }
"""

"""
function that defines the ODE of the seesaw logic gate
"""


def f(y, t, k, kf):
    return np.array([
        - k * y[0] * y[3] + k * y[4] * y[7],
        - k * y[1] * y[3] + k * y[5] * y[7],
        - k * y[2] * y[3] + k * y[6] * y[7],
        - k * y[0] * y[3] + k * y[4] * y[7] - k * y[1] * y[3] + k * y[5] * y[7] - k * y[2] * y[3] + k * y[6] * y[7],
        k * y[0] * y[3] - k * y[4] * y[7],
        k * y[1] * y[3] - k * y[5] * y[7],
        k * y[2] * y[3] - k * y[6] * y[7],
        k * y[0] * y[3] - k * y[4] * y[7] + k * y[1] * y[3] - k * y[5] * y[7] + k * y[2] * y[3] - k * y[6] * y[7] - kf *
        y[7] * y[9] - k * y[7] * y[13] + k * y[14] * y[8] + k * y[15] * y[14] - k * y[16] * y[7],
        k * y[7] * y[13] - k * y[14] * y[8],
        - kf * y[7] * y[9],
        kf * y[7] * y[9],
        kf * y[7] * y[9],
        k * y[8] * y[17],
        - k * y[7] * y[13] + k * y[14] * y[8],
        k * y[7] * y[13] - k * y[14] * y[8] - k * y[15] * y[14] + k * y[16] * y[7],
        - k * y[15] * y[14] + k * y[16] * y[7],
        k * y[15] * y[14] - k * y[16] * y[7],
        - k * y[8] * y[17],
        k * y[8] * y[17],
    ])


def main():
    file_name_template = r'final_proj{0:02d}.pdf'

    # true false input values
    initial_inputs = [(0, 0, 0),
                      (0, 0, 1),
                      (0, 1, 0),
                      (1, 0, 0),
                      (1, 1, 0),
                      (0, 1, 1),
                      (1, 0, 1),
                      (1, 1, 1)]

    # initialized arrays
    out_val_array = []
    sim_length = 36000
    ts = np.linspace(0, sim_length, sim_length + 1)
    idx = 0

    for (x, y, z) in initial_inputs:
        init_conc = 1

        # initialized values
        x_0, y_0, z_0, eo_0 = x * init_conc, y * init_conc, z * init_conc, 0
        sum_0, ig1_0, ig2_0, ig3_0 = 3 * init_conc, 0, 0, 0
        f_0 = 2 * init_conc
        th_0 = 1.5 * init_conc
        sg_0, sgout_0 = 1 * init_conc, 0
        rep_0 = 1.5 * init_conc
        w1_0, w2_0 = 0, 0
        t1_0, t2_0, t3_0 = 0, 0, 0
        gf_0 = 0

        """
              0    1    2     3    4      5      6      7     8     9    10    11    12    13     14      15    16   17   18
        y = {[x], [y], [z], [s], [ig1], [ig2], [ig3], [w1], [w2], [th], [t1], [t2], [t3], [sg], [sgout], [f], [gf], [r], [eo]}
        """
        y0 = np.array(
            [x_0, y_0, z_0, sum_0, ig1_0, ig2_0, ig3_0, w1_0, w2_0, th_0, t1_0, t2_0, t3_0, sg_0, sgout_0, f_0,
             gf_0, rep_0, eo_0])

        # reaction rates
        k = 5e-4  # 5e5 /M/s = 5e-4 /nM/s
        kf = 5e-2  # 5e-2 /nM/s

        # integrate
        res = integrate.odeint(f, y0, ts, args=(k, kf))

        # getting output values
        out_vals = res[:, 18]

        # individual input plots
        plt.plot(ts, out_vals, linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Output')
        plt.title('x={}, y={}, z={}'.format(x, y, z))
        plt.savefig(file_name_template.format(idx), bbox_inches='tight')
        plt.clf()
        idx += 1

        out_val_array.append(out_vals)

    # putting all of the outs on one plot
    jet = plt.get_cmap('jet')
    colors = iter(jet(np.linspace(0, 1, 9)))
    for i in range(len(out_val_array)):
        plt.plot(ts, out_val_array[i], color=next(colors), linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Output')
    plt.title('All 8 Outputs')
    plt.text(40000, 1.5, '(1 v 1) ^ 1', fontsize=10)
    plt.text(40000, 1.3, '(1 v 0) ^ 1', fontsize=10)
    plt.text(40000, 1.2, '(1 v 1) ^ 0', fontsize=10)
    plt.text(40000, 1.1, '(0 v 1) ^ 1', fontsize=10)
    plt.text(40000, 0.4, '(1 v 0) ^ 0', fontsize=10)
    plt.text(40000, 0.2, '(0 v 0) ^ 1', fontsize=10)
    plt.text(40000, 0.3, '(0 v 1) ^ 0', fontsize=10)
    plt.text(40000, 0.0, '(0 v 0) ^ 0', fontsize=10)
    plt.savefig(file_name_template.format(10), bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    main()
