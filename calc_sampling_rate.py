"""
计算基分类器的采样率






"""

import math

IR = 3
n = 15
delta_P = 1/(IR*math.log(IR, 2))
delta_N = abs(1/(IR*math.log(IR, 2)) - 1)
r_P = []
r_N = []
for i in range(1, n+1):
    r_i_p = 1/IR - ((2**(i-n))*delta_P)
    r_P.append(r_i_p)
    r_i_n = 1/IR + ((2**(i-n))*delta_N)
    r_N.append(r_i_n)

print("IR:", IR)
print("r_P:")
for r in r_P:
    print(r, end="\t")
print()
print("r_N:")
for r in r_N:
    print(r, end="\t")

