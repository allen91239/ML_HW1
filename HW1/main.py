import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.linalg import inv

x = list()
y = list()

def main():
    global n, lam
    n = int(input())
    lam = int(input())
    f = open("testcase.txt", "r")
    if f.mode == 'r':
        content = f.readlines()
    read_data(content)
    lse()
    newton()
    f.close()

# seperate into x[i], y[i]
def read_data(content):
    for i in content:
        temp = i.split(',')
        x.append(temp[0])
        if(temp[1][-1] == "\n"):
            temp[1] = temp[1][:-1]
        y.append(temp[1])

def LU_inverse_implement(mat):
    # find LU then use forward and backward substitution
    # crout LU algo https://www.gamedev.net/tutorials/programming/math-and-physics/matrix-inversion-using-lu-decomposition-r3637/
    temp_mat = mat
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    #for i in range(n):
    #    L[i][0] = temp_mat[i][0]
    #for j in range(1,n):
    #    U[0][j] = temp_mat[0][j] / L[0][0]
    for j in range(n):
        U[j][j] = 1
        for i in range(j, n):
            sum = 0
            for k in range(j):
                sum = sum + L[i][k]*U[k][j]
            L[i][j] = temp_mat[i][j] - sum
        for k in range(j+1, n):
            sum0 = 0
            for i in range(j):
                sum0 = sum0 + L[j][i]*U[i][k]
            U[j][k] = (temp_mat[j][k] - sum0) / L[j][j]
    #sum1 = 0
    #for k in range(n-1):
    #    sum1 = sum1 + L[n-1][k]*U[k][n-1]
    #L[n-1][n-1] = temp_mat[n-1][n-1] - sum1
    """
    for i in range(n-1):
        for j in range(i+1, n):
            factor = -(temp_mat[i][i])
            L[j][i] = -(temp_mat[j][i]/factor)
            for k in range(i, n):
                temp_mat[j][k] = temp_mat[j][k] + temp_mat[i][k]*temp_mat[j][i]/factor
    U = temp_mat
    """

    out = list()
    inverse_mat = np.zeros((n,n))

    for i in range(n):
        y = [0 for i in range(n)]
        x = [0 for i in range(n)]
        row = np.zeros(n)
        row[i] = 1
        for j in range(n):
            forward = 0
            for k in range(j):
                forward = forward + L[j][k]*y[k]
            y[j] = (row[j] - forward) / L[j][j]
        for j in range(n-1, -1, -1):
            backward = 0
            for k in range(j+1, n):
                backward = backward + U[j][k] * x[k]
            x[j] = (y[j] - backward) / U[j][j]
        out.append(x)
    inverse_mat = np.transpose(out)

    return inverse_mat

def lse():
    #matrix
    mat = np.zeros((len(x), n))
    mat_t = np.zeros((n, len(x)))
    y_mat = np.zeros((len(y), 1))
    lam_mat = np.identity(n)
    for i in range(n):
        lam_mat[i][i] = lam
    for i in range(len(y)):
        y_mat[i][0] = y[i]
    for i in range(len(x)):
        for j in range(n):
            mat[i][j] = pow( float(x[i]), j)
            mat_t[j][i] = mat[i][j]

    coeffs = LU_inverse_implement(np.add((mat.transpose().dot(mat)), lam_mat)).dot(mat.transpose()).dot(y_mat)
    formula = str()
    error = 0
    holder = 0
    for i in range(len(x)):
        for j in range(n):
            holder = holder + coeffs[j]*pow(float(x[i]), j)
        error = error + pow( (holder - float(y[i])), 2)
        holder = 0
    for i in range(n):
        if (i == n-1):
            formula = formula + str(str(coeffs[i]) + "X^" + str(i))
        else:
            formula = formula + str(str(coeffs[i]) + "X^" + str(i)) + " + "
    formula = formula.replace('[', '')
    formula = formula.replace(']', '')
    strerror = str(error)
    strerror = strerror.replace('[', '')
    strerror = strerror.replace(']', '')
    print("LSE:")
    print("Fitting line: " + formula)
    print("Total error: " + strerror + "\n")
    x_data = list()
    y_data = list()

    for i in range(len(x)):
        x_data.append(float(x[i]))
    for j in range(len(y)):
        y_data.append(float(y[j]))
    plt.scatter(x_data, y_data, c='r')

    x_out = list()
    x_out = np.linspace(min(x_data), max(x_data), 100)
    y_out = list()
    for i in range(len(x_out)):
        result = 0
        for j in range(n):
            result = result + coeffs[j]*x_out[i]**j
        y_out.append(float(result))
    plt.plot(x_out, y_out)
    plt.savefig("lse.png")
    plt.clf()

def newton():
    mat = np.zeros((len(x), n))
    mat_t = np.zeros((n, len(x)))
    y_mat = np.zeros((len(y), 1))
    lam_mat = np.identity(n)
    for i in range(n):
        lam_mat[i][i] = lam
    for i in range(len(y)):
        y_mat[i][0] = y[i]
    for i in range(len(x)):
        for j in range(n):
            mat[i][j] = pow( float(x[i]), j)
            mat_t[j][i] = mat[i][j]
    a = np.zeros(n)
    a = np.reshape(a, (n, 1))
    for i in range(3):
        gradient = np.subtract( (mat_t.dot(mat).dot(a)).dot(2), mat_t.dot(y_mat).dot(2))
        hessian = mat_t.dot(mat).dot(2)
        a = np.subtract(a, LU_inverse_implement(hessian).dot(gradient))
    coeffs = a
    error = 0
    holder = 0
    formula = str()
    for i in range(len(x)):
        for j in range(n):
            holder = holder + coeffs[j]*pow(float(x[i]), j)
        error = error + pow( (holder - float(y[i])), 2)
        holder = 0
    for i in range(n):
        if (i == n-1):
            formula = formula + str(str(coeffs[i]) + "X^" + str(i))
        else:
            formula = formula + str(str(coeffs[i]) + "X^" + str(i)) + " + "
    formula = formula.replace('[', '')
    formula = formula.replace(']', '')
    strerror = str(error)
    strerror = strerror.replace('[', '')
    strerror = strerror.replace(']', '')
    print("Newton:")
    print("Fitting line: " + formula)
    print("Total error: " + strerror + "\n")
    x_data = list()
    y_data = list()

    for i in range(len(x)):
        x_data.append(float(x[i]))
    for j in range(len(y)):
        y_data.append(float(y[j]))
    plt.scatter(x_data, y_data, c='r')

    x_out = list()
    x_out = np.linspace(min(x_data), max(x_data), 100)
    y_out = list()
    for i in range(len(x_out)):
        result = 0
        for j in range(n):
            result = result + coeffs[j]*x_out[i]**j
        y_out.append(float(result))
    plt.plot(x_out, y_out)
    plt.savefig("newton.png")
    plt.clf()
    
#def visualize():


if __name__ == "__main__":
    main()
