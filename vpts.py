import torch as pt
from matplotlib import pyplot as plt
from rc import RC
from error import Error

device = 'cuda' if pt.cuda.is_available() else 'cpu'
pt.set_default_dtype(pt.float64)
seed = 16
pt.manual_seed(seed)
dim_x = 1
N = 10
T0, T1 = 50, 100
T_max = 200
l_All_VPTs = []
s_All_VPTs = []
h_All_VPTs = []
T2 = T1 + T_max 
sample_size = 100
r_0 = pt.zeros((N, 1), device = device)

s_rho = 0.8353366916477455
s_sparsity = 0.96
s_sigma = 1.5641523755434774
s_bias = 7.422063288375644e-06
u = 0.955

for x_0 in pt.rand(sample_size):
    s_x_t = pt.zeros((dim_x, T2 + 1), device = device)
    s_x_t[0, 0] = x_0
    VPT = 0
    for i in range(T2):
        s_x_t[0, i + 1] = u * pt.sin(pt.pi * s_x_t[0, i])
    s_x_std = pt.std(s_x_t[:, T0 + 1:T1 + 1])

    for t in range(T_max):
        s_model = RC(device, seed, dim_x, N, s_rho, s_sparsity, s_sigma, s_bias)
        s_model.train(s_x_t[:, :T0 + 1], s_x_t[:, T0 + 1:T1 + 1], r_0)
        s_res_pred, s_q_res_pred = s_model.reservoir_prediction(t)
        s_res_pred_err = Error(s_x_t[:, T1:T1 + t + 1], s_res_pred)
        RMSE = s_res_pred_err.RMSE()
        if(RMSE < s_x_std):
            VPT = VPT + 1
        else:
            s_All_VPTs.append(VPT)
            break
print("Mean VPT Sine :", sum(s_All_VPTs) / len(s_All_VPTs))


h_rho = 0.6074166603943667
h_sparsity = 0.94
h_sigma = 0.35371283527605013
h_bias = -0.5996761514948162
a, b = 1.4, 0.3

for x_0 in pt.rand((sample_size, 2)):
    h_x_t = pt.zeros((dim_x, T2 + 1), device = device)
    h_x_t[0, :2] = x_0
    VPT = 0
    for i in range(1, T2):
        h_x_t[0, i + 1] = 1 - a * h_x_t[0, i] ** 2 + b * h_x_t[0, i - 1]
    h_x_std = pt.std(h_x_t[:, T0 + 1:T1 + 1])

    for t in range(T_max):
        h_model = RC(device, seed, dim_x, N, h_rho, h_sparsity, h_sigma, h_bias)
        h_model.train(h_x_t[:, :T0 + 1], h_x_t[:, T0 + 1:T1 + 1], r_0)
        h_res_pred, h_q_res_pred = h_model.reservoir_prediction(t)
        h_res_pred_err = Error(h_x_t[:, T1:T1 + t + 1], h_res_pred)
        RMSE = h_res_pred_err.RMSE()
        if(RMSE < h_x_std):
            VPT = VPT + 1
        else:
            h_All_VPTs.append(VPT)
            break
print("Mean VPT henon :", sum(h_All_VPTs) / len(h_All_VPTs))


l_rho = 0.866350431222052
l_sparsity = 0.94
l_sigma = 1.2667028204891229
l_bias = 8.680828069765185e-05
r = 3.91

for x_0 in pt.rand(sample_size):
    l_x_t = pt.zeros((dim_x, T2 + 1), device = device)
    l_x_t[0, 0] = x_0
    VPT = 0
    for i in range(T2):
        l_x_t[0, i + 1] = r * l_x_t[0, i] * (1 - l_x_t[0, i])
    l_x_std = pt.std(l_x_t[:, T0 + 1:T1 + 1])

    for t in range(T_max):
        l_model = RC(device, seed, dim_x, N, l_rho, l_sparsity, l_sigma, l_bias)
        l_model.train(l_x_t[:, :T0 + 1], l_x_t[:, T0 + 1:T1 + 1], r_0)
        l_res_pred, l_q_res_pred = l_model.reservoir_prediction(t)
        l_res_pred_err = Error(l_x_t[:, T1:T1 + t + 1], l_res_pred)
        RMSE = l_res_pred_err.RMSE()
        if(RMSE < l_x_std):
            VPT = VPT + 1
        else:
            l_All_VPTs.append(VPT)
            break
print("Mean VPT Logistic :", sum(l_All_VPTs) / len(l_All_VPTs))


plt.figure(figsize=(5, 5))
plt.boxplot([l_All_VPTs, s_All_VPTs, h_All_VPTs], patch_artist=True, boxprops=dict(facecolor='skyblue'))
plt.title('Distribution of Valid Prediction Times')
plt.xlabel('Dynamical System')
plt.xticks([1, 2, 3], ['Logistic Map', 'Sine Map', 'Hénon Map'])
plt.ylabel('Valid Prediction Time')
plt.show()