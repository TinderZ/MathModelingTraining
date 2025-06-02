'''
Author: Zhurun Zhang
Date: 2025-04-19 00:10:52
LastEditors: Zhurun Zhang
LastEditTime: 2025-04-19 01:39:11
FilePath: \国赛训练3\prove1.py
Description: Always happy to chat! Reach out via email < b23042510@njupt.edu.cn or 2857895300@qq.com >

'''
import numpy as np

# Define the matrix A
A = np.array([
    [ 0, 10,  4, 50,  5,  0,  2],
    [ 0, 10, 30,  5, 12,  0,  0],
    [20,  0,  0,  0,  0,  5,  3],
    [ 8,  0,  0,  0,  0,  6, 10],
    [ 0,  6,  5, 10, 11,  4,  0],
    [25, 40, 60, 120, 40, 11, 15],
    [60, 70, 120, 140, 80, 25, 55]
])

A1 = A[:6]
A2 = A[[0, 1, 2, 3, 4, 6]]
# Define the vector b
b_vec = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [6.327047],
    [13.79802]
])

b1 = b_vec[:6]
b2 = b_vec[[0, 1, 2, 3, 4, 6]]

# Solve the system Ax = b for x using least squares
# lstsq returns a tuple: (solution, residuals, rank, singular_values)
solution1, residuals1, rank1, s1 = np.linalg.lstsq(A1, b1, rcond=None)

# The solution vector x contains the values for a, b, c, d, e, f, g
a, b, c, d, e, f, g = solution1.flatten()

print("Least-squares solution for 1(a, b, c, d, e, f, g):")
print(f"a1 = {a}")
print(f"b1 = {b}")
print(f"c1 = {c}")
print(f"d1 = {d}")
print(f"e1 = {e}")
print(f"f1 = {f}")
print(f"g1 = {g}")

# Optionally, you can check the norm of the residuals to see how well the solution fits
if len(residuals1) > 0:
    print(f"\nNorm of the residuals (||Ax - b||): {np.sqrt(residuals1[0])}")
else:
    print("\nNo residuals returned (system may be consistent)")