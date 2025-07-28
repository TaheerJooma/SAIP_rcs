import torch as pt

class Error:
    
    def __init__(self, d_actual, d_pred):
        self.d_actual = d_actual
        self.d_pred = d_pred
        
    # Mean Absolute Error
    def MAE(self):
        diff = self.d_actual - self.d_pred
        abs_diff = pt.abs(diff)
        tot_sum = pt.sum(abs_diff)
        err = tot_sum / (self.d_actual.shape[0] * self.d_actual.shape[1])
        return err
    
    # Mean Square Error
    def MSE(self):
        diff = self.d_actual - self.d_pred
        sq_diff = pt.square(diff)
        tot_sum = pt.sum(sq_diff)
        err = tot_sum / (self.d_actual.shape[0] * self.d_actual.shape[1])
        return err
        
    # Root Mean Square Error
    def RMSE(self):
        mse = self.MSE() 
        err = pt.sqrt(mse)
        return err
    
    def Error_Analysis(self):
        print('MAE = {:.8f}'.format(self.MAE()))
        print('MSE = {:.8f}'.format(self.MSE()))
        print('RMSE = {:.8f}'.format(self.RMSE()))