import torch as pt
import numpy as np
from rc import RC

def lyap_exp_1D(f, f_deriv, x, n_trans=10000, n_iter=100000):
    for n in range(n_trans):
        x = f(x)
    result = []
    for n in range(n_iter):
        result.append(np.log(abs(f_deriv(x))))
        x = f(x)
    return np.mean(result)


class Lyp_Spectrum:
    
    def __init__(self, rc_model):
        self.dim = rc_model.dim_r
        self.W_i = rc_model.W_i.cpu().numpy()
        self.W_r = rc_model.W_r.cpu().numpy()
        self.bias = rc_model.bias.cpu().numpy()
        self.W_o = rc_model.W_o.cpu().numpy()
        self.r_T1 = rc_model.r_T1.cpu().numpy()
    
    def __r_next(self, r_t):
        r_t = r_t.reshape((self.dim, 1))
        r_next = np.tanh(self.W_r @ r_t + self.W_i @ (self.W_o @ r_t) + self.bias)
        return r_next
    
    def __jac_res_map(self, r_t):
        r_t = r_t.reshape((self.dim, 1))
        main_diag = 1 - (np.tanh(self.W_r @ r_t + self.W_i @ (self.W_o @ r_t) + self.bias))**2
        jac = np.diag(main_diag[:, 0]) @ (self.W_r + self.W_i @ self.W_o)
        return jac

    def comp_lyap_spectr(self, n_trans=10000, n_iter=100000, dim=None, f=None, J=None, x=None):
        if any(var is None for var in [dim, f, J, x]):
            dim = self.dim
            f = self.__r_next
            J = self.__jac_res_map
            x = self.r_T1    
        for n in range(n_trans):
            x = f(x)
        V = np.eye(dim)
        lyap_sums = np.zeros(dim)
        for n in range(n_iter):
            V = J(x) @ V
            Q, R = np.linalg.qr(V)
            lyap_sums += np.log(np.abs(np.diag(R)))
            V = Q
            x = f(x)
        lyap_exps = lyap_sums / n_iter
        return lyap_exps        
    

device = 'cuda' if pt.cuda.is_available() else 'cpu'
pt.set_default_dtype(pt.float64)
np.set_printoptions(suppress = True, precision = 8)
seed = 16
dim_x = 1
N = 10
T0, T1, T2 = 50, 100, 100
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
l_rc_lyap_spec = Lyp_Spectrum(l_model).comp_lyap_spectr()
l_rc_lyap_max_exp = np.max(l_rc_lyap_spec)
l_act_lyap_exp = lyap_exp_1D(lambda l: r * l * (1 - l), lambda l: r * (1 - 2 * l), l_x_0)
l_abs_err_lyap_max_exp = np.abs(l_act_lyap_exp - l_rc_lyap_max_exp)
print('Logistic map RC Lyapunov spectrum:')
print(l_rc_lyap_spec)
print('Logistic map RC maximal Lyapunov Exponent:')
print('{:.8f}'.format(l_rc_lyap_max_exp))
print('Approximate actual Logistic map Lyapunov Exponent:')
print('{:.8f}'.format(l_act_lyap_exp))
print('Logistic MAE of actual vs predicted maximal Lyapunov Exponent:')
print('{:.8f}'.format(l_abs_err_lyap_max_exp))
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
s_rc_lyap_spec = Lyp_Spectrum(s_model).comp_lyap_spectr()
s_rc_lyap_max_exp = np.max(s_rc_lyap_spec)
s_act_lyap_exp = lyap_exp_1D(lambda s: u * np.sin(np.pi * s), lambda s: np.pi * u * np.cos(np.pi * s), s_x_0)
s_abs_err_lyap_max_exp = np.abs(s_act_lyap_exp - s_rc_lyap_max_exp)
print('Sine map RC Lyapunov spectrum:')
print(s_rc_lyap_spec)
print('Sine map RC maximal Lyapunov Exponent:')
print('{:.8f}'.format(s_rc_lyap_max_exp))
print('Approximate actual Sine map Lyapunov Exponent:')
print('{:.8f}'.format(s_act_lyap_exp))
print('Sine MAE of actual vs predicted maximal Lyapunov Exponent:')
print('{:.8f}'.format(s_abs_err_lyap_max_exp))
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
h_lyap_spec = Lyp_Spectrum(h_model)
h_rc_lyap_spec = h_lyap_spec.comp_lyap_spectr()
h_rc_lyap_max_exp = np.max(h_rc_lyap_spec)
h_map = lambda h: np.array([[1 - a * h[0, 0]**2 + h[1, 0]], [b * h[0, 0]]])
h_jac = lambda h: np.array([[-2 * a * h[0, 0], 1], [b, 0]])
h_act_lyap_exp = h_lyap_spec.comp_lyap_spectr(dim=2, f=h_map, J=h_jac, x=np.zeros((2, 1)))
h_abs_err_lyap_max_exp = np.abs(np.max(h_act_lyap_exp) - h_rc_lyap_max_exp)
h_abs_err_lyap_sec_exp = np.abs(h_act_lyap_exp[1] - h_rc_lyap_spec[1])
print('Henon map RC Lyapunov spectrum:')
print(h_rc_lyap_spec)
print('Henon map RC maximal Lyapunov Exponent:')
print('{:.8f}'.format(h_rc_lyap_max_exp))
print('Approximate actual Henon map Lyapunov spectrum:')
print(h_act_lyap_exp)
print('Approximate actual Henon map maximal Lyapunov Exponent:')
print('{:.8f}'.format(np.max(h_act_lyap_exp)))
print('Henon MAE of actual vs predicted maximal Lyapunov Exponent:')
print('{:.8f}'.format(h_abs_err_lyap_max_exp))
print('Henon MAE of actual vs predicted secondary Lyapunov Exponent:')
print('{:.8f}'.format(h_abs_err_lyap_sec_exp))