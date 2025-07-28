from scipy.optimize import fsolve
import numpy as np
import torch as pt
from rc import RC
import warnings

class FP_Analysis:
    # res_eqs : reservoir computer model
    # r_vars : r variables
    # tol : tolerance for fixed point search
    # trunc : truncation for comparsion of doubles
    def __init__(self, rc_model, tol, trunc):
        self.dim = rc_model.dim_r
        self.W_i = rc_model.W_i.cpu().numpy()
        self.W_r = rc_model.W_r.cpu().numpy()
        self.bias = rc_model.bias.cpu().numpy()
        self.W_o = rc_model.W_o.cpu().numpy()
        self.tol = tol
        self.trunc = trunc
          
    def __fp_eqs(self, r_t):
        r_t = r_t.reshape((self.dim, 1))
        eq_res = np.tanh(self.W_r @ r_t + self.W_i @ (self.W_o @ r_t) + self.bias)
        eq_fp = eq_res - r_t
        return eq_fp[:, 0]
    
    def __jac_res_map(self, r_t):
        r_t = r_t.reshape((self.dim, 1))
        main_diag = 1 - (np.tanh(self.W_r @ r_t + self.W_i @ (self.W_o @ r_t) + self.bias))**2
        jac = np.diag(main_diag[:, 0]) @ (self.W_r + self.W_i @ self.W_o)
        return jac

    def find_r_fp(self, num_sims):
        all_fps = np.ones((self.dim, 1))
        # solve for fixed point numerically
        for i in range(num_sims):
            init_guess = np.random.rand(self.dim, 1)
            fp_pos = fsolve(self.__fp_eqs, init_guess, xtol=self.tol)
            if(np.allclose(self.__fp_eqs(fp_pos), np.zeros((self.dim, 1)))):
                    all_fps = np.column_stack((all_fps, np.array(fp_pos).astype(np.float64)))
            fp_neg = fsolve(self.__fp_eqs, -init_guess, xtol=self.tol)
            if(np.allclose(self.__fp_eqs(fp_neg), np.zeros((self.dim, 1)))):
                    all_fps = np.column_stack((all_fps, np.array(fp_neg).astype(np.float64)))
        # check if solultion lies in the set |r| < 1
        rem_ind = []
        for i in range(all_fps.shape[1]):
            col = all_fps[:, i].round(self.trunc)
            if np.any(((col <= -1) | (col >= 1))) is np.True_:
                rem_ind.append(i)
        all_fps = np.delete(all_fps, tuple(rem_ind), axis=1)
        # collect all unique solutions
        all_fps = all_fps[:, np.unique(all_fps.round(self.trunc), axis=1, return_index=True)[1]]
        return all_fps
    
    def pred_x_fp(self, r_fps):
        return self.W_o @ r_fps
    
    def find_abs_eigval_J_fps(self, r_fp):
        jac_fp = self.__jac_res_map(r_fp)
        eigvals_J_fp = np.linalg.eigvals(jac_fp)
        return np.abs(eigvals_J_fp)
        
    def fp_r_stability(self, r_fps):
        stability = []
        for i in range(r_fps.shape[1]):
            abs_eigval_J_fp = self.find_abs_eigval_J_fps(r_fps[:, i])
            # absolute values of all eigenvalues of J(fp) <= 1 -> stable, else unstable
            if np.any(abs_eigval_J_fp > 1) is np.True_:
                stability.append('unstable')
            else:
                stability.append('stable')
        return stability  
    

warnings.filterwarnings('ignore', category=RuntimeWarning)
device = 'cuda' if pt.cuda.is_available() else 'cpu'
np.set_printoptions(suppress = True, precision = 8)
pt.set_default_dtype(pt.float64)
seed = 16
np.random.seed(seed)
dim_x = 1
N = 10
T0, T1, T2 = 50, 100, 100
r_0 = pt.zeros((N, 1), device = device)


h_rho = 0.6074166603943667
h_sparsity = 0.95
h_sigma = 0.35371283527605013
h_bias = -0.5996761514948162
h_x_0 = 0
h_x_t = pt.zeros((dim_x, T2 + 1), device = device)
h_x_t[0, 0] = h_x_0
a, b = 1.4, 0.3
h_act_fp1 = (-7 + np.sqrt(609)) / 28 
h_act_fp2 = (-7 - np.sqrt(609)) / 28
for i in range(T2):
   h_x_t[0, i + 1] = 1 - a * h_x_t[0, i] ** 2 + b * h_x_t[0, i - 1]
   
h_model = RC(device, seed, dim_x, N, h_rho, h_sparsity, h_sigma, h_bias)
h_model.train(h_x_t[:, :T0 + 1], h_x_t[:, T0 + 1:T1 + 1], r_0)
h_fp_analysis = FP_Analysis(h_model, 1e-10, 8)
h_r_fps = h_fp_analysis.find_r_fp(10000)
h_x_fps = h_fp_analysis.pred_x_fp(h_r_fps)
h_stab = h_fp_analysis.fp_r_stability(h_r_fps)
print('Henon map reservoir fixed points:')
print(h_r_fps)
print('Henon map predicted fixed points:')
print(h_x_fps)
print('Henon map predicted stability of fixed points:')
print(h_stab)
print('Henon map MAE of actual vs predicted fixed points :')
print('{:.8f}'.format(np.abs(h_act_fp1 - h_x_fps[0, 0])))
print('{:.8f}'.format(np.abs(h_act_fp2 - h_x_fps[0, 1])))
print()


l_rho = 0.866350431222052
l_sparsity = 0.94
l_sigma = 1.2667028204891229
l_bias = 8.680828069765185e-05
l_x_0 = 0.63
l_x_t = pt.zeros((dim_x, T2 + 1), device = device)
l_x_t[0, 0] = l_x_0
r = 3.91
l_act_fp1 = 0.0
l_act_fp2 = 291 / 391
for i in range(T2):
    l_x_t[0, i + 1] = r * l_x_t[0, i] * (1 - l_x_t[0, i])

l_model = RC(device, seed, dim_x, N, l_rho, l_sparsity, l_sigma, l_bias)
l_model.train(l_x_t[:, :T0 + 1], l_x_t[:, T0 + 1:T1 + 1], r_0)
l_fp_analysis = FP_Analysis(l_model, 1e-10, 8)
l_r_fps = l_fp_analysis.find_r_fp(10000)
l_x_fps = l_fp_analysis.pred_x_fp(l_r_fps)
l_stab = l_fp_analysis.fp_r_stability(l_r_fps)
print('Logistic map reservoir fixed points:')
print(l_r_fps)
print('Logistic map predicted fixed points:')
print(l_x_fps)
print('Logistic map predicted stability of fixed points:')
print(l_stab)
print('Logistic map MAE of actual vs predicted fixed points :')
print('{:.8f}'.format(np.abs(l_act_fp1 - l_x_fps[0, 1])))
print('{:.8f}'.format(np.abs(l_act_fp2 - l_x_fps[0, 2])))

print()


s_rho = 0.8353366916477455
s_sparsity = 0.96
s_sigma = 1.5641523755434774
s_bias = 7.422063288375644e-06
s_x_0 = 0.71
s_x_t = pt.zeros((1, T2 + 1), device = device)
s_x_t[0] = s_x_0
u = 0.955
s_act_fp1 = fsolve(lambda s: u * np.sin(np.pi * s) - s, 0.1, xtol=1e-10)
s_act_fp2 = fsolve(lambda s: u * np.sin(np.pi * s) - s, 0.5, xtol=1e-10)
for i in range(T2):
    s_x_t[0, i + 1] = u * pt.sin(pt.pi * s_x_t[0, i])

s_model = RC(device, seed, dim_x, N, s_rho, s_sparsity, s_sigma, s_bias)
s_model.train(s_x_t[:, :T0 + 1], s_x_t[:, T0 + 1:T1 + 1], r_0)
s_fp_analysis = FP_Analysis(s_model, 1e-10, 8)
s_r_fps = s_fp_analysis.find_r_fp(10000)
s_x_fps = s_fp_analysis.pred_x_fp(s_r_fps)
s_stab = s_fp_analysis.fp_r_stability(s_r_fps)
print('Sine map reservoir fixed points:')
print(s_r_fps)
print('Sine map predicted fixed points:')
print(s_x_fps)
print('Sine map predicted stability of fixed points:')
print(s_stab)
print('Sine map MAE of actual vs predicted fixed points :')
print('{:.8f}'.format(np.abs(s_act_fp1 - s_x_fps[0, 2])[0]))
print('{:.8f}'.format(np.abs(s_act_fp2 - s_x_fps[0, 1])[0]))