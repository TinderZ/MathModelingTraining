import numpy as np
from GM_model import DGM21, GM11

x0 = np.array([87321, 87850, 78193, 73107, 61220, 69455, 70491, 69047, 63387])
c = 0
x0 = x0 + c
pred = DGM21(x0, 3) - c
print(pred)

for i in range(1, 9):
    o1 = np.exp(-2 / (i + 1))
    o2 = np.exp(2 / (i + 2))
    oo = x0[i-1] / x0[i]
    print(o1, oo, o2)
# x0 = np.array([71.1, 72.4, 72.4, 72.1, 71.4, 72, 71.6])
# pred2 = GM11(x0, 0)
# print(pred2)
