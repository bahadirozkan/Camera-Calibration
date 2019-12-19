#!/usr/bin/env python
# coding: utf-8
#Computer Vision HW2, Bahadır Özkan

import pandas as pd
import numpy as np
from numpy import linalg as LA

def main():
    df = pd.read_csv('data.csv', delimiter=";")

    R_exact = [[0.7472, 0.0401, 0.6634],
               [0.5417, 0.5417,-0.6428],
               [-0.3851,0.8396, 0.3830]]
    T_exact = [120, 10, 13]
    alpha_exact = 1
    f_exact = 100
    K_exact =[[f_exact, 0, 0],
                 [0, f_exact, 0],
                 [0, 0, 1]]

    df["u"] = 1.0e+04 * df["u"]
    df["v"] = 1.0e+04 * df["v"]

    L = len(df["X"])

    A = np.zeros((L, 8), dtype=np.float64)
    #print("Shape of Matrix A : ", A.shape)

    for i in range(L):
        X, Y, Z = df["X"][i], df["Y"][i], df["Z"][i]
        x, y = df["u"][i], df["v"][i]

        A[i] = np.array([x*X, x*Y, x*Z, x, -y*X, -y*Y, -y*Z, -y],dtype=np.float64)

    #print(A)

    u, s, vh = np.linalg.svd(A)
    #print("Shape of U, s and Vh:", u.shape, s.shape, vh.shape)

    """
    Zc is greater than 0 hence signs of r1 and r2 is reversed
    https://www.cse.unr.edu/~bebis/CS791E/Notes/CameraCalibration.pdf
    """
    v = vh[np.argmin(s)]
    v = -1*v
    #print(v)

    gamma = np.linalg.norm([v[0], v[1], v[2]])
    print('gamma:',gamma)

    v = v / gamma

    r1 = [v[4], v[5], v[6]]
    alpha = np.linalg.norm(r1)
    print('alpha:',alpha)

    r1 = r1 / alpha
    #print('r1:',r1)

    # r2
    r2 = [v[0], v[1], v[2]]
    #print('r2:', r2)

    r3 = np.cross(r1, r2)
    #print('r3:', r3)

    R = [r1, r2, r3]
    print("R:",R)

    frames = [df["X"],df["Y"],df["Z"]]
    XYZ = pd.concat(frames,axis=1)

    b = df['u'] * XYZ.dot(r3).T

    T_x = v[7]/alpha
    T_y = v[3]

    A_prime = np.array([-df['u'], (XYZ.dot(r1).T + T_x)])

    res= np.linalg.lstsq(A_prime.T,b,rcond=None)

    T_z = res[0][0]
    f_x = res[0][1]

    T = np.array([T_x, T_y, T_z])

    K = np.array([[f_x, 0, 0],
                [0, f_x, 0],
                [0, 0, 1]]);

    print("T:",T)
    print("K:",K)

    #Performance results
    #Norm diffrences of T and R values show that the exact and calculated values are close to each other.
    #Whereas, there is a slight diffence on the exact and calculated K values.
    print("Frobenius norm difference between T_exact and calculated T:", LA.norm(T_exact) - LA.norm(T))
    print("Frobenius norm difference between K_exact and calculated K:", LA.norm(K_exact) - LA.norm(K))
    print("Frobenius norm difference between R_exact and calculated R:", LA.norm(R_exact) - LA.norm(R))

if __name__ == "__main__": main()
