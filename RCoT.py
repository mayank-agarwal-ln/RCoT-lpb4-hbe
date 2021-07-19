from matrix2 import matrix2
from random_fourier import random_fourier_features
from normalize import normalize
from dist import dist
from expandgrid import expandgrid
import numpy as np
import pandas as pd
import scipy.linalg as sc

# ' RCoT - tests whether x and y are conditionally independent given z. Calls RIT if z is empty.
# ' @param x Random variable x.
# ' @param y Random variable y.
# ' @param z Random variable z.
# ' @param approx Method for approximating the null distribution. Options include:
# ' "lpd4," the Lindsay-Pilla-Basak method (default),
# ' "gamma" for the Satterthwaite-Welch method,
# ' "hbe" for the Hall-Buckley-Eagleson method,
# ' "chi2" for a normalized chi-squared statistic,
# ' "perm" for permutation testing (warning: this one is slow but recommended for small samples generally <500 )
# ' @param num_f Number of features for conditioning set. Default is 25.
# ' @param num_f2 Number of features for non-conditioning sets. Default is 5.
# ' @param seed The seed for controlling random number generation. Use if you want to replicate results exactly. Default is NULL.
# ' @return A list containing the p-value \code{p} and statistic \code{Sta}
# ' @export
# ' @examples
# ' RCIT(rnorm(1000),rnorm(1000),rnorm(1000));
# '
# ' x=rnorm(10000);
# ' y=(x+rnorm(10000))^2;
# ' z=rnorm(10000);
# ' RCIT(x,y,z,seed=2);
# def cov(X,Y):
#   N = X.shape[0]
#   c = np.sum((X-np.mean(X))*(Y-np.mean(Y)))
#   return c/(N-1)

def RCoT(x, y, z=None, approx="lpd4", num_f=5, num_f2=5, seed=None):
    

    x = np.matrix(x).T
    
    y = np.matrix(y).T

    # Unconditional Testing
    #   if (length(z)==0):
    #     out=RIT(x,y,approx=approx,seed=seed);
    #     return(out)
    
    z = np.matrix(z).T 
    
    # else:
    # else:

    x = matrix2(x)
    # print("x: ",x)
    y = matrix2(y)
    # print("y: ",y)
    z = matrix2(z)
    # print("z: ",z)

    # z = z[i for i in range(z.shape[1]) if z[:,i].std() > 0]
    # Convert later to lamnda function
    z1 = []
    try:
        c = z.shape[1]
    except:
        c = 1

    for i in range(c):
        if(z[:, i].std() > 0):
            z1.append(z[:, i])

    z = z1[0]
    z = matrix2(z)
    # print("z1: ",z)
    try:
        d = z.shape[1]    # D => dimension of variable
    except:
        d = 1
    # print("d: ",d)
    # Unconditional Testing
    # if (length(z)==0):
    #   out=RIT(x,y,approx=approx, seed=seed);
    #   return(out);

    # Sta - test statistic -> s
    # if sd of x or sd of y == 0 then x and y are independent
    if (x.std() == 0 or y.std() == 0):
      # p=1 and Sta=0
        out = (1, 0)
        return(out)

    r = x.shape[0]
    # print(r)
    if (r > 500):
        r1 = 500
    else:
        r1 = r

    # Normalize = making it as mean =0 and std= 1
    x = normalize(x).T
    # print("x: ",x)
    y = normalize(y).T
    # print("y: ",y)
    z = normalize(z).T
    # print("z: ",z)

    (four_z, w, b) = random_fourier_features(
        z[:, :d], num_f=num_f, sigma=np.median(dist(z[:r1, ])), seed=seed)
    # print("z")
    # print("w")
    # print(w)
    # print("b")
    # print(b)
    # print("four_z")
    # print(four_z)
    (four_x, w, b) = random_fourier_features(
        x, num_f=num_f2, sigma=np.median(dist(x[:r1, ])), seed=seed)
    # print("x")
    # print("w")
    # print(w)
    # print("b")
    # print(b)
    # print("four_x")
    # print(four_x)
    (four_y, w, b) = random_fourier_features(
        y, num_f=num_f2, sigma=np.median(dist(y[:r1, ])), seed=seed)
    # print("y")
    # print("w")
    # print(w)
    # print("b")
    # print(b)
    # print("four_y")
    # print(four_y)
    # print(b)
    # print("four_z: ",four_z)
    # print("four_x: ",four_x)
    # print("four_y: ",four_y)
    # What is this required for?
    f_x = normalize(four_x)
    f_y = normalize(four_y)  # n,numf2
    f_z = normalize(four_z)  # n,numf
    print("f_x: ",f_x)
    print()
    print("f_y: ",f_y)
    print()
    print("f_z: ",f_z)
    print()

    # Next few lines will be Equation2 from RCoT paper
    Cxy = np.cov(f_x, f_y, rowvar=False)  # 2*numf2,2*numf2
    # print("Cxy: ", Cxy)
    # Cx = np.cov(f_x, rowvar=False)
    # print(Cx == Cxy[:num_f2,:num_f2])
    Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2
    # print(np.diag(Cxy))
    # print("Cxy: ",Cxy)
    # Cxy = cov(f_x, f_y)  # 2*numf2,2*numf2
    # Cxy = Cxy[:num_f2, num_f2:]  # num_f2,num_f2
    # Cxy = Cxy[:num_f2, :num_f2]
    Czz = np.cov(f_z,rowvar=False)  # numf,numf
    # Czz = cov(f_z,f_z)  # numf,numf
    # print(np.mean(Czz))
    print("Cxy: ",Cxy)
    print()
    print("Czz: ",Czz)
    print()
    # print("Czz: ",Czz)
    # for i in range(2*num_f2):
    #     for j in range(2*num_f2):
    #         print(Cxy[i][j], end=" ")
    #     print()

    I = np.eye(num_f)
    L = sc.cholesky((Czz + (np.eye(num_f) * 1e-10)), lower=True)
    L_inv = sc.solve_triangular(L, I, lower=True)
    i_Czz = L_inv.T.dot(L_inv)  # numf,numf
    print("i_Czz: ",i_Czz)

    Cxz = np.cov(f_x, f_z, rowvar=False)[:num_f2, num_f2:]  # numf2,numf
    # Cxz = cov(f_x, f_z)[:num_f2, num_f2:]
    # Cxz = np.cov(f_x, f_z, rowvar=False)[:num_f2, :num_f]
    print("Cxz: ",Cxz)
    print()
    # Czy = np.cov(f_z, f_y, rowvar=False)[:num_f, num_f:]  # numf,numf2
    Czy = np.cov(f_z, f_y, rowvar=False)[:num_f, :num_f2]
    # Czy = np.cov(f_z, f_y)[:num_f, :num_f2]
    print("Czy: ", Czy)
    print()

    # z_i_Czz, e_x_z, e_y_z  ?????
    # z_i_Czz = f_z @ i_Czz  # (n,numf) * (numf,numf)
    # e_x_z = z_i_Czz @ Cxz.T  # n,numf
    # e_y_z = z_i_Czz @ Czy

    # approximate null distributions

    # residual of fourier after it removes the effect of ??
    # res_x = f_x-e_x_z
    # res_y = f_y-e_y_z

    # if (num_f2 == 1):
        # approx = "hbe"

    # if (approx == "perm"):
    #   pass

    # else:
        # Cross covariance of XY.Z
    matmul = (Cxz @ (i_Czz @ Czy))
    print("matrixmul: ",matmul)
    Cxy_z = Cxy-matmul  # less accurate for permutation testing
    # Equation26 calculating S' in RCoT paper
    Sta = r * np.sum(Cxy_z**2)


    # ncx = np.linspace(1,f_x.shape[1],f_x.shape[1])
    # ncy = np.linspace(1,f_y.shape[1],f_y.shape[1])
    # ncx = list(ncx)
    # ncy = list(ncy)
    # d = expandgrid(ncx,ncy)
    # res = res_x[:,d['Var1']] * res_y[:,d['Var2']]
    # Cov = 1/r * (res.T @ res)

    # w,v = np.linalg.eigh(Cov)
    # w = [i for i in w if i>0]

    # # if(approx == "lpbd4"):
    # w1 = w
    # p=try(1-lpb4(eig_d_values,Sta),silent=TRUE);
    # if (!is.numeric(p) | is.nan(p)){
    #    p=1-hbe(eig_d$values,Sta);
    # }
    

    return (Cxy_z, Sta)


# z = np.linspace(1, 100, 100)
# y = z*2
# x = z + y

data = pd.read_csv("M2.csv")
# print(data)
x = data['A']
z = data['B']
y = data['C']
# print(x)
# print(y)
# print(z)
(Cxy_z, Sta) = RCoT(x, y, z, seed = 8)
print("Cxy_z: ",Cxy_z)
# print()
print("Sta: ",Sta)