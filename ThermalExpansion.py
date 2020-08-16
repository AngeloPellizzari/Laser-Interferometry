# importing the necessary modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

# calling the file with the data to be plotted and reading it in
fname = 'DataW3T2.csv'
data = np.loadtxt(fname, delimiter=',', comments='#', usecols=(0, 1, 2))

# defining the time, potential, and temperature data
traw = data[:, 0]
vraw = data[:, 1]
temp_raw = data[:, 2]

# defining constants for given data
lam = 632.8E-9  # m wavelength of laser
l0 = 0.0882  # m initial length of copper
ul0 = 0.0001  # m uncertainty in initial length of copper


ind_max_v = argrelextrema(vraw, np.greater)
print(ind_max_v)
n = np.size(ind_max_v)
print(n)
dT = temp_raw[0] - temp_raw[-1]
udT = 0.05

#print(dT)
dl = n * lam / 2
a = dl / (l0 * dT)
udl = 0.05e-9

print(a)
# storing the a values calculated by running each data set through this script
aCu = 16.5e-6
a1 = 21.625862371232147e-06
a2 = 24.530921196258008e-06
a3 = 34.43928617467333e-06
a4 = 28.099554674948628e-06
a5 = 32.11116824502477e-06
a6 = 30.79344694585332e-06

ua1 = 0.004281906879537995
ua2 = 0.0043707658040539835
ua3 = 0.004470631594197897
ua4 = 0.0043612748365218406
ua5 = 0.0041198522652016595
ua6 = 0.0042798390322429615

ua = np.sqrt((udT / dT)**2 + (ul0 / l0)**2 + (udl/dl))
print(ua)


# defining function to calculate the dl values over the changes in temperature
def l_data(a_val, T_vals):
    dl_vals = a_val * (l0 * T_vals)
    return dl_vals

T_range = np.arange(21.5, 50.5, 0.5)
dl0 = l_data(aCu, T_range) + l0
dl1 = l_data(a1, T_range) + l0
dl2 = l_data(a2, T_range) + l0
dl3 = l_data(a3, T_range) + l0
dl4 = l_data(a4, T_range) + l0
dl5 = l_data(a5, T_range) + l0
dl6 = l_data(a6, T_range) + l0

# defining the packing factor for packing large amounts of data
npac = 1


# defining the pack function which takes an array of data and a packing factor
# and packs the data by the packing factor
def pack(A, p):
    B = np.zeros(len(A) // p)
    i = 1
    while i * p < len(A):
        B[i - 1] = np.mean(A[p * (i - 1):p * i])
        i += 1
    return B


# packing the data
t = pack(traw, npac)
v = pack(vraw, npac)
temp = pack(temp_raw, npac)

# defining the uncertainty in y.
sigmay = 0

# plotting the data
plt.plot(T_range, dl0, 'k-', label='aCu')
plt.plot(T_range, dl1, 'b-', label='a1')
plt.plot(T_range, dl2, 'g-', label='a2')
plt.plot(T_range, dl3, 'r-', label='a3')
plt.plot(T_range, dl4, 'c-', label='a4')
plt.plot(T_range, dl5, 'm-', label='a5')
plt.plot(T_range, dl6, 'y-', label='a6')
#plt.errorbar(traw, temp_raw, yerr=sigmay)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.title('Trial 6: Temperature vs Time')
plt.legend(loc = 'upper left', ncol = 2, prop = {'size':9})
plt.show()

plt.errorbar(traw, vraw, yerr=sigmay)
plt.xlabel('Time [s]')
plt.ylabel('Potential [J]')
plt.title('Trial 6: Potential vs Time')
plt.show()

plt.errorbar(T_range, dl1)
plt.show()



print(type(ind_max_v[0]))