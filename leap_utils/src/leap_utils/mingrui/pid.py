import numpy as np


class PIDController:
    def __init__(self, n_dim, delta_t):
        self.n_dim = n_dim
        self.delta_t = delta_t
        self.last_error = np.zeros((n_dim, 1))
        self.accumulated_error = np.zeros((n_dim, 1))
        self.b_set_gains = False

    def setGains(self, Kp, Kv, Ki):
        Kp = np.array(Kp)
        Kv = np.array(Kv)
        Ki = np.array(Ki)
        if Kp.size == 1:
            Kp = Kp * np.eye(self.n_dim)
        if Kv.size == 1:
            Kv = Kv * np.eye(self.n_dim)
        if Ki.size == 1:
            Ki = Ki * np.eye(self.n_dim)
        self.Kp, self.Kv, self.Ki = Kp, Kv, Ki
        self.b_set_gains = True

    def controlInput(self, curr_err):
        if not self.b_set_gains:
            raise NameError("The PID control gains have not been set.")
        curr_err = curr_err.reshape(-1, 1)
        delta_error = (curr_err - self.last_error) / self.delta_t
        self.last_error = curr_err
        self.accumulated_error += curr_err * self.delta_t

        control_input = self.Kp @ curr_err + self.Kv @ delta_error + self.Ki @ self.accumulated_error
        return control_input

    def reset(self):
        self.last_error = np.zeros((self.n_dim, 1))
        self.accumulated_error = np.zeros((self.n_dim, 1))
