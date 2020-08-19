import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete
from scipy.linalg import expm
import matplotlib.pyplot as plt

data = loadmat('F16traindata_CMabV_2020.mat')
z_k = np.transpose(data['Z_k'])
c_m = data['Cm'][:, 0]
u_k = np.transpose(data['U_k'])

alpha_m = z_k[0,:]
beta_m = z_k[1,:]
v_m = z_k[2,:]

Au = u_k[0,:]
Av = u_k[1,:]
Aw = u_k[2,:]


def plot_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90,0)
    ax.plot_trisurf(alpha_m, beta_m, c_m, cmap=cm.jet)
    ax.set_xlabel(r'$\alpha_m$')
    ax.set_ylabel(r'$\beta_m$')
    ax.set_zlabel(r'$V_{tot}$')
    plt.show()

#
# def rk4_step(f, x, t, dt, *args):
#     """
#     One step of 4th Order Runge-Kutta method
#     """
#     k = dt
#     k1 = k * f(t, x, *args)
#     k2 = k * f(t + 0.5*k, x + 0.5*k1, *args)
#     k3 = k * f(t + 0.5*k, x + 0.5*k2, *args)
#     k4 = k * f(t + dt, x + k3, *args)
#     return x + 1/6. * (k1 + 2*k2 + 2*k3 + k4)
#
#
# def rk4(f, t, x0, *args):
#     """
#     4th Order Runge-Kutta method
#     """
#     n = len(t)
#     x = np.zeros((n, len(x0)))
#     x[0] = x0
#     for i in range(n-1):
#         dt = t[i+1] - t[i]
#         x[i+1] = rk4_step(f, x[i], t[i], dt, *args)
#     return x


def rk4(f, x, u, t):
    """
    4th Order Runge-Kutta method
    """
    h = (t[1] - t[0]) / 2
    N = (t[1] - t[0]) / h

    for j in range(0, int(N)):
        k1 = h * f(x, u, t[0] + j * h)
        k2 = h * f(x + k1 / 2, u, t[0] + j * h + h / 2)
        k3 = h * f(x + k2 / 2, u, t[0] + j * h + h / 2)
        k4 = h * f(x + k3, u, t[0] + j * h + h)

        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return [t, x]


def f(x, u, t):
    """
    System dynamics
    """
    x_dot = np.zeros(len(x))
    np.put(x_dot, [0, 1, 2], u)
    return x_dot


def h(x):
    """
    Output dynamics
    """
    u = x[0]
    v = x[1]
    w = x[2]
    c_alpha_up = x[3]
    z_pred = np.zeros((3))

    z_pred[0] = np.arctan2(w,u) * (c_alpha_up+1)
    z_pred[1] = np.arctan2(v, np.sqrt(u**2 + w**2))
    z_pred[2] = np.sqrt(u**2 + v**2 + w**2)
    return z_pred


# def F(x, u, t):
#     """
#     Calculate Jacobian of f
#     """
#     return np.zeros([4, 4])


def jacobian_h(t, x, u):
    #     """
    #     Calculate Jacobian of h, H
    #     """
    jacobian = np.zeros((3,4))
    u = x[0]
    v = x[1]
    w = x[2]
    c_alpha_up = x[3]
    jacobian[0,0] = -(c_alpha_up + 1) * w/(u**2+w**2)
    jacobian[0,1] = 0.
    jacobian[0,2] = (c_alpha_up + 1) * u/(u**2+w**2)
    jacobian[0,3] = np.arctan2(w,u)
    jacobian[1,0] = -((u*v)/(np.sqrt(u**2 + w**2) * (u**2 + v**2 + w**2)))
    jacobian[1,1] = np.sqrt(u**2 + w**2)/(u**2 + v**2 + w**2)
    jacobian[1,2] = -((v*w)/(np.sqrt(u**2 + w**2) * (u**2 + v**2 + w**2)))
    jacobian[1,3] = 0.
    jacobian[2,0] = u / np.sqrt(u**2 + v**2 + w**2)
    jacobian[2,1] = v / np.sqrt(u**2 + v**2 + w**2)
    jacobian[2,2] = w / np.sqrt(u**2 + v**2 + w**2)
    jacobian[2,3] = 0.
    return jacobian


def c2d(a, b, t):  # todo; make own c2d function
    """Continuous to discrete, based on MATLAB code"""
    n  = np.size(a, 1)
    nb = np.size(b, 1)
    temp1 = np.concatenate((a, b), axis=1)*t
    temp2 = np.zeros((nb,n+nb))

    temp = np.concatenate((temp1, temp2), 0)

    s = expm(temp)
    Phi = s[0:n,0:n]
    Gamma = s[0:n,n:n+nb]
    return Phi, Gamma


dt = 0.01  # time step
N = len(c_m)  # data length
n = 4  # number of states
nm = 3  # number of measurements
m = 3  # number of inputs

B = np.vstack((np.identity(3), np.zeros(3)))  # input matrix
G = np.identity(n)  # noise input matrix

sigma_w = [1E-3, 1E-3, 1E-3, 0]
Q = np.diag(np.square(sigma_w))

sigma_v = [0.035, 0.013, 0.110]
R = np.diag(np.square(sigma_v))

sigma_x0 = [2.0, 2.0, 2.0, 50.0]
P_0 = np.diag(np.square(sigma_x0))  # initial guess

Ex_0 = [150., 0., 0., 0.]  # initial guess

# =========================

XX_k1k1 = np.zeros((n, N))
PP_k1k1 = np.zeros((n, N))
SIGMA_x_cor = np.zeros((n, N))
z_pred = np.zeros((nm, N))
IEKFitcount = np.zeros(N)
# epsilon = 1e-12
# doIEKF = 1
# do_plot = 0
# maxIterations = 2
T = np.zeros((1,N))

x_k_1k_1 = Ex_0  # assign initial guess
P_k_1k_1 = P_0  # assign initial guess

ti = 0
tf = dt

U_k = u_k
Z_k = z_k

for k in range(0, N):
    # One-step ahead prediction
    [t, x_kk_1] = rk4(f, x_k_1k_1, U_k[:,k], [ti, tf])
    T[:,k] = t[0]  # alternatively; take the center value?

    z_kk_1 = h(x_kk_1)
    z_pred[:,k] = z_kk_1

    # Calculate Jacobians
    F = np.zeros([4, 4])
    H = jacobian_h(0, x_kk_1, U_k[:, k])

    # Discretize state transition & input matrices
    [dummy, Psi] = c2d(F, B, dt)
    [Phi, Gamma] = c2d(F, G, dt)

    # Cov. matrix of state pred. error
    P_kk_1 = Phi * P_k_1k_1 * Phi.T + Gamma * Q * Gamma.T
    sigma_x_pred = np.sqrt(np.diag(P_kk_1))

    # Kalman gain calculation
    K = P_kk_1.dot(H.T).dot(np.linalg.inv(H.dot(P_kk_1).dot(H.T) + R))

    # Measurement update
    # x_k_1k_1 = x_kk_1 + K.dot(z_kk_1 - h(x_kk_1))
    x_k_1k_1 = x_kk_1 + K.dot(Z_k[:, k] - z_kk_1)
    # todo; iterated

    # Cov. matrix of state estimation error
    P_k_1k_1 = (np.identity(n) - K.dot(H)).dot(P_kk_1)
    sigma_x_cor = np.sqrt(np.diag(P_k_1k_1))
    SIGMA_x_cor[:, k] = sigma_x_cor

    # Time step
    ti = np.round(tf, 2)
    tf = np.round(tf + dt, 2)

    # todo; check rank of observation matrix, must be full rank, for sensor fusion
    # todo; analysis of state observability. Derivative of derivative.
    # todo; define matrix using sympy Matrix([udot])

    XX_k1k1[:, k] = x_k_1k_1
# print(XX_k1k1[:, -1])
do_plot = 1
if do_plot:
    start = 0
    end = -1

    # plt.close("all")
    # plt.plot(XX_k1k1[0,:]-X_k[0,:])

    # plt.figure(dpi=300)
    # plt.title(r'$XX_{k1k1}$')
    # plt.plot(T[0, start:end], np.transpose(XX_k1k1[0, start:end]), label=r'$u$')
    # plt.plot(T[0, start:end], np.transpose(XX_k1k1[1, start:end]), label=r'$v$')
    # plt.plot(T[0, start:end], np.transpose(XX_k1k1[2, start:end]), label=r'$w$')
    # plt.plot(T[0, start:end], np.transpose(XX_k1k1[3, start:end]), label=r'$C_{\alpha_{up}}$')
    # plt.xlabel(r'$Time\,[s]$')
    # plt.legend()
    # plt.show()
    #
    # plt.figure(dpi=300)
    # plt.title(r'$STD_{cross-correlation}$')
    # plt.plot(T[0, 0:500], np.transpose(SIGMA_x_cor[0, 0:500]), label=r'$u$')
    # plt.plot(T[0, 0:500], np.transpose(SIGMA_x_cor[1, 0:500]), label=r'$v$')
    # plt.plot(T[0, 0:500], np.transpose(SIGMA_x_cor[2, 0:500]), label=r'$w$')
    # plt.plot(T[0, 0:500], np.transpose(SIGMA_x_cor[3, 0:500]), label=r'$C_{\alpha_{up}}$')
    # plt.ylim([-0.5, 3.5])
    # plt.xlabel(r'$Time\,[s]$')
    # plt.ylabel(r'$\beta\,[rad]$')
    # plt.legend()
    # plt.show()
    #
    plt.figure(dpi=300)
    plt.title(r'$C_{\alpha_{up}}$')
    plt.plot(T[0, start:end], XX_k1k1[3, start:end])
    plt.xlabel(r'$Time\,[s]$')
    plt.ylabel(r'$C_{\alpha_{up}}\,[-]$')
    # plt.legend()
    plt.show()

    plt.figure(dpi=300)
    plt.title(r'Measured vs Predicted vs Corrected $\alpha$')
    plt.plot(T[0, start:end], np.transpose(z_k[0, start:end]), color='tab:blue', label=r'$\alpha_m$')
    plt.plot(T[0, start:end], np.transpose(z_pred[0, start:end]), color='tab:orange', linewidth=2, label=r'$\alpha_p$')
    plt.plot(T[0,start:end], np.transpose(z_k[0,start:end]-XX_k1k1[3,start:end]), color='tab:green', label=r'$\alpha_{corrected}$')  # todo; check if correct
    plt.xlabel(r'$Time\,[s]$')
    plt.ylabel(r'$\alpha\,[rad]$')
    plt.ylim(bottom=-1, top=1)

    # # polyfit
    # DEGREE = 2
    # V_poly, res, _, _, _ = np.polyfit(T[0, start:end], np.transpose(z_pred[0, start:end]), DEGREE, DEGREE, full=True)
    # print(res / len(T[0, start:end]))
    # V_poly_f = np.poly1d(V_poly)
    # plt.plot(V_poly_f(np.arange(0, 100, 1)), label='polyfit')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(dpi=300)
    plt.title(r'Measured vs Predicted $\beta$')
    plt.plot(T[0, start:end], np.transpose(z_k[1, start:end]), color='tab:blue', label=r'$\beta_m$')
    plt.plot(T[0, start:end], np.transpose(z_pred[1, start:end]), color='tab:orange', linewidth=2, label=r'$\beta_p$')
    plt.xlabel(r'$Time\,[s]$')
    plt.ylabel(r'$\beta\,[rad]$')

    # # polyfit
    # DEGREE = 2
    # V_poly, res, _, _, _ = np.polyfit(T[0, start:end], np.transpose(z_pred[1, start:end]), DEGREE, DEGREE, full=True)
    # print(res / len(T[0, start:end]))
    # V_poly_f = np.poly1d(V_poly)
    # plt.plot(V_poly_f(np.arange(0, 100, 1)), label='polyfit')


    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(dpi=300)
    plt.title(r'Measured vs Predicted $V$')
    plt.plot(T[0, start:end], np.transpose(z_k[2, start:end]), color='tab:blue', label=r'$V_m$')
    plt.plot(T[0, start:end], np.transpose(z_pred[2, start:end]), color='tab:orange', linewidth=2, label=r'$V_p$')
    plt.xlabel(r'$Time\,[s]$')
    plt.ylabel(r'$V\,[rad]$')

    # # polyfit
    # DEGREE = 2
    # V_poly, res, _, _, _ = np.polyfit(T[0, start:end], np.transpose(z_pred[2, start:end]), DEGREE, full=True)
    # print(res/len(T[0, start:end]))
    # V_poly_f = np.poly1d(V_poly)
    # plt.plot(V_poly_f(np.arange(0, 100, 1)), label='polyfit')
    plt.grid()
    plt.legend()
    plt.show()


    # plt.figure(dpi=300)
    # plt.title('Measured data')
    # plt.plot(T[0, start:end], alpha_m[start:end])
    # plt.show()


# generate train, val, test data
PERCENTAGE_VALIDATION = 0.2
np.random.seed(1)

alpha_corrected = z_k[0,0:-1]-XX_k1k1[3,0:-1]  # todo; this or c_p
beta_p = z_pred[1, 0:-1]
cm_a = XX_k1k1[3, 0:-1]

data = np.array([alpha_corrected, beta_p, cm_a])
np.savetxt("all_data.csv", data, delimiter=",")
time_sequence_indices = np.arange(5000, int(5000+10/0.01))
time_sequence = data[:, time_sequence_indices]
np.savetxt("time_sequence.csv", time_sequence, delimiter=",")

indices = np.random.permutation(data[0].shape[0])
training_idx, test_idx = indices[:int((1-PERCENTAGE_VALIDATION)*len(alpha_corrected))], indices[int((1-PERCENTAGE_VALIDATION)*len(alpha_corrected)):]
training, test = data[:,training_idx], data[:,test_idx]

np.savetxt("data/train.csv", training, delimiter=",")
np.savetxt("data/test.csv", test, delimiter=",")

