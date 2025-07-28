import torch as pt
from matplotlib import pyplot as plt
from rc import RC
from error import Error

device = 'cuda' if pt.cuda.is_available() else 'cpu'
pt.set_default_dtype(pt.float64)
seed = 16
dim_x = 1
N = 10
T0, T1, T2 = 50, 100, 150
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
l_test_train = l_model.test_training()
l_one_step_pred, l_q_one_step_pred = l_model.one_step_prediction(l_x_t[:, T1:])
l_res_pred, l_q_res_pred = l_model.reservoir_prediction(T2 - T1)

l_train_err = Error(l_x_t[:, T0 + 1:T1 + 1], l_test_train)
print('Logistic Map Training Errors')
l_train_err.Error_Analysis()
l_one_step_pred_err = Error(l_x_t[:, T1:], l_one_step_pred)
print('Logistic Map One Step Prediction Errors')
l_one_step_pred_err.Error_Analysis()
l_res_pred_err = Error(l_x_t[:, T1:], l_res_pred)
print('Logistic Map Reservoir Prediction Errors')
l_res_pred_err.Error_Analysis()

l_x_t = l_x_t.cpu()
l_test_train = l_test_train.cpu()
l_one_step_pred = l_one_step_pred.cpu()
l_res_pred = l_res_pred.cpu()

print()

s_rho = 0.8353366916477455
s_sparsity = 0.96
s_sigma = 1.5641523755434774
s_bias = 7.422063288375644e-06
s_x_0 = 0.71
s_x_t = pt.zeros((1, T2 + 1), device = device)
s_x_t[0] = s_x_0
u = 0.955
for i in range(T2):
    s_x_t[0, i + 1] = u * pt.sin(pt.pi * s_x_t[0, i])

s_model = RC(device, seed, dim_x, N, s_rho, s_sparsity, s_sigma, s_bias)
s_model.train(s_x_t[:, :T0 + 1], s_x_t[:, T0 + 1:T1 + 1], r_0)
s_test_train = s_model.test_training()
s_one_step_pred, s_r_one_step_pred = s_model.one_step_prediction(s_x_t[:, T1:])
s_res_pred, r_res_pred = s_model.reservoir_prediction(T2 - T1)

s_train_err = Error(s_x_t[:, T0 + 1:T1 + 1], s_test_train)
print('Sine Map Training Errors')
s_train_err.Error_Analysis()
s_one_step_pred_err = Error(s_x_t[:, T1:], s_one_step_pred)
print('Sine Map One Step Prediction Errors')
s_one_step_pred_err.Error_Analysis()
s_res_pred_err = Error(s_x_t[:, T1:], s_res_pred)
print('Sine Map Reservoir Prediction Errors')
s_res_pred_err.Error_Analysis()

s_x_t = s_x_t.cpu()
s_test_train = s_test_train.cpu()
s_one_step_pred = s_one_step_pred.cpu()
s_res_pred = s_res_pred.cpu()

print()

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
h_test_train = h_model.test_training()
h_one_step_pred, h_r_one_step_pred = h_model.one_step_prediction(h_x_t[:, T1:].to(device))
h_res_pred, h_r_res_pred = h_model.reservoir_prediction(T2 - T1)

h_train_err = Error(h_x_t[:, T0 + 1:T1 + 1], h_test_train)
print('Henon Map Training Errors')
h_train_err.Error_Analysis()
h_one_step_pred_err = Error(h_x_t[:, T1:], h_one_step_pred)
print('Henon Map One Step Prediction Errors')
h_one_step_pred_err.Error_Analysis()
h_res_pred_err = Error(h_x_t[:, T1:], h_res_pred)
print('Henon Map Reservoir Prediction Errors')
h_res_pred_err.Error_Analysis()

h_x_t = h_x_t.cpu()
h_test_train = h_test_train.cpu()
h_one_step_pred = h_one_step_pred.cpu()
h_res_pred = h_res_pred.cpu() 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4), layout='constrained')

ax1.set_title('Logistic map')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$x_{t}$')
ax1.plot(time[T1:], l_x_t[0, T1:], label='actual')
ax1.plot(time[T1:], l_one_step_pred[0, :], linestyle='--', color='red', label='predicted')

ax2.set_title('Sine map')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$x_{t}$')
ax2.plot(time[T1:], s_x_t[0, T1:])
ax2.plot(time[T1:], s_one_step_pred[0, :], linestyle='--', color='red')

ax3.set_title('Hénon map')
ax3.set_xlabel('$t$')
ax3.set_ylabel('$x_{t}$')
ax3.plot(time[T1:], h_x_t[0, T1:])
ax3.plot(time[T1:], h_one_step_pred[0, :], linestyle='--', color='red')

fig.suptitle('One-step predictions', fontsize='16')
fig.legend(loc='outside lower center', ncol=2)
plt.show()


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4), layout='constrained')

ax1.set_title('Logistic map')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$x_{t}$')
ax1.plot(time[T1:], l_x_t[0, T1:], label='actual')
ax1.plot(time[T1:], l_res_pred[0, :], linestyle='--', color='red', label='predicted')

ax2.set_title('Sine map')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$x_{t}$')
ax2.plot(time[T1:], s_x_t[0, T1:])
ax2.plot(time[T1:], s_res_pred[0, :], linestyle='--', color='red')

ax3.set_title('Hénon map')
ax3.set_xlabel('$t$')
ax3.set_ylabel('$x_{t}$')
ax3.plot(time[T1:], h_x_t[0, T1:])
ax3.plot(time[T1:], h_res_pred[0, :], linestyle='--', color='red')

fig.suptitle('Reservoir predictions', fontsize='16')
fig.legend(loc='outside lower center', ncol=2)
plt.show()
