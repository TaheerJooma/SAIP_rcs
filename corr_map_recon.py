import torch as pt
from matplotlib import pyplot as plt
from rc import RC

device = 'cuda' if pt.cuda.is_available() else 'cpu'
pt.set_default_dtype(pt.float64)
seed = 16
dim_x = 1
N = 10
T0, T1, T2 = 50, 100, 10100
time = pt.arange(0, T2 + 1, 1)
r_0 = pt.zeros((N, 1), device = device)

l_rho = 0.866350431222052
l_sparsity = 0.94
l_sigma = 1.2667028204891229
l_bias = 8.680828069765185e-05
l_x_0 = 0.63
l_x_t = pt.zeros((dim_x, T2 + 1), device = device)
l_x_t[0, 0] = l_x_0
r = 3.91
for i in range(T2):
    l_x_t[0, i + 1] = r * l_x_t[0, i] * (1 - l_x_t[0, i])

l_model = RC(device, seed, dim_x, N, l_rho, l_sparsity, l_sigma, l_bias)
l_model.train(l_x_t[:, :T0 + 1], l_x_t[:, T0 + 1:T1 + 1], r_0)
l_res_pred, l_q_res_pred = l_model.reservoir_prediction(T2 - T1)
l_x_act = pt.column_stack((l_x_t[0, T1 : T2 - 1], l_x_t[0, T1 + 1 : T2]))
l_x_pred = pt.column_stack((l_res_pred[0, :(T2 - T1)], l_res_pred[0, 1:]))


s_rho = 0.7963488749673642
s_sparsity = 0.931
s_sigma = 1.790048794288635
s_bias = 0.0
s_x_0 = 0.71
s_x_t = pt.zeros((1, T2 + 1), device = device)
s_x_t[0] = s_x_0
u = 0.955
for i in range(T2):
    s_x_t[0, i + 1] = u * pt.sin(pt.pi * s_x_t[0, i])

s_model = RC(device, seed, dim_x, N, s_rho, s_sparsity, s_sigma, s_bias)
s_model.train(s_x_t[:, :T0 + 1], s_x_t[:, T0 + 1:T1 + 1], r_0)
s_res_pred, r_res_pred = s_model.reservoir_prediction(T2 - T1)
s_x_act = pt.column_stack((s_x_t[0, T1 : T2 - 1], s_x_t[0, T1 + 1 : T2]))
s_x_pred = pt.column_stack((s_res_pred[0, :(T2 - T1)], s_res_pred[0, 1:]))


h_rho = 0.6074166603943667
h_sparsity = 0.95
h_sigma = 0.35371283527605013
h_bias = -0.5996761514948162
h_x_0 = 0
h_x_t = pt.zeros((dim_x, T2 + 1), device = device)
h_x_t[0, 0] = h_x_0
a, b = 1.4, 0.3
for i in range(T2):
   h_x_t[0, i + 1] = 1 - a * h_x_t[0, i] ** 2 + b * h_x_t[0, i - 1]
   
h_model = RC(device, seed, dim_x, N, h_rho, h_sparsity, h_sigma, h_bias)
h_model.train(h_x_t[:, :T0 + 1], h_x_t[:, T0 + 1:T1 + 1], r_0)
h_res_pred, h_r_res_pred = h_model.reservoir_prediction(T2 - T1)
h_x_act = pt.column_stack((h_x_t[0, T1 + 1 : T2], h_x_t[0, T1 : T2 - 1]))
h_x_pred = pt.column_stack((h_res_pred[0, 1:], h_res_pred[0, :(T2 - T1)]))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,4), layout='constrained')

ax1.set_title('logistic map')
ax1.set_xlabel('$x_t$')
ax1.set_ylabel('$x_{t+1}$')
ax1.scatter(l_x_act[:, 0].cpu(), l_x_act[:, 1].cpu(), s=3)
ax1.scatter(l_x_pred[:, 0].cpu(), l_x_pred[:, 1].cpu(), color='red', s=3, alpha=0.7)
x = pt.linspace(0, 1, 100)
y = r * x * (1 - x)
ax1.plot(x, y, label='${}x (1 - x)$'.format(r), color='green', alpha=0.3)

ax2.set_title('sine map')
ax2.set_xlabel('$x_t$')
ax2.set_ylabel('$x_{t+1}$')
ax2.scatter(s_x_act[:, 0].cpu(), s_x_act[:, 1].cpu(), s=3)
ax2.scatter(s_x_pred[:, 0].cpu(), s_x_pred[:, 1].cpu(), color='red', s=3, alpha=0.7)
x = pt.linspace(0, 1, 100)
y = u * pt.sin(pt.pi * x)
ax2.plot(x, y, label='${} \sin(\pi x)$'.format(u), color='purple', alpha=0.3)

ax3.set_title('Hénon map')
ax3.set_xlabel('$x_t$')
ax3.set_ylabel('$x_{t+1}$')
ax3.scatter(h_x_act[:, 0].cpu(), h_x_act[:, 1].cpu(), marker='.', label='actual', s=3)
ax3.scatter(h_x_pred[:, 0].cpu(), h_x_pred[:, 1].cpu(), marker='.', color='red', label='predicted', s=3)

fig.suptitle('Correlation map reconstruction', fontsize='16')
fig.legend(loc='outside lower center', ncol=2)
plt.show()