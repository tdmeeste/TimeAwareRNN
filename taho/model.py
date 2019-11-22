import torch
import torch.nn as nn
import math



def RK(x0, y, f, dt, scheme, x_half=None, x_full=None):
    # explicit Runge Kutta methods
    # scheme in ['Euler', 'Midpoint', 'Kutta3', 'RK4']
    # x0 = x(t_n); optional x_half = x(t + 0.5 * dt), x_full = x(t + dt);
    # if not present, x0 is used (e.g. for piecewise constant inputs).

    if scheme == 'Euler':
        incr = dt * f(x0, y)
    elif scheme == 'Midpoint':
        x1 = x0 if x_half is None else x_half
        k1 = f(x0, y)
        k2 = f(x1, y + dt * (0.5 * k1))  # x(t_n + 0.5 * dt)
        incr = dt * k2
    elif scheme == 'Kutta3':
        x1 = x0 if x_half is None else x_half
        x2 = x0 if x_full is None else x_full
        k1 = f(x0, y)
        k2 = f(x1, y + dt * (0.5 * k1))  # x(t_n + 0.5 * dt)
        k3 = f(x2, y + dt * (- k1 + 2 * k2))  # x(t_n + 1.0 * dt)
        incr = dt * (k1 + 4 * k2 + k3) / 6
    elif scheme == 'RK4':
        x1 = x0 if x_half is None else x_half
        x2 = x0 if x_half is None else x_half
        x3 = x0 if x_full is None else x_full
        k1 = f(x0, y)
        k2 = f(x1, y + dt * (0.5 * k1))  # x(t_n + 0.5 * dt)
        k3 = f(x2, y + dt * (0.5 * k2))  # x(t_n + 0.5 * dt)
        k4 = f(x3, y + dt * k3)  # x(t_n + 1.0 * dt)
        incr = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        raise NotImplementedError

    return y + incr



class GRUCell(nn.Module):
    """
    simple baseline: standard nn.GRUCell with weight drop
    ignore all init arguments except for k_in, k_out, k_state, dropout
    """
    def __init__(self, k_in, k_out, k_state, dropout=0., **kwargs):
        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.state_size = k_state

        #self.cell = nn.GRUCell(k_in, k_state)
        self.expand_input = nn.Sequential(nn.Linear(k_in, k_state), nn.Tanh())
        self.cell = nn.GRUCell(k_state, k_state)
        self.Ly = nn.Linear(k_state, k_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        #TODO: add regularization to current baseline
        #TODO: change output regression tool

        #self.init_params()


    def init_params(self):
        stdv = 1.0 / math.sqrt(self.k_state)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, state, **kwargs):
        state_new = self.dropout(self.cell(self.expand_input(x), state))
        y_new = self.Ly(state_new)
        return y_new, state_new





class HOGRUCell(nn.Module):
    """
    higher order GRU cell; 1st order with equidistant samples is equivalent to standard GRU cell
    """
    def __init__(self, k_in, k_out, k_state, dropout=0., meandt=1, train_scheme='Euler', eval_scheme='same', **kwargs):

        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.meandt = meandt
        self.state_size = k_state
        self.train_scheme = train_scheme
        self.eval_scheme = eval_scheme

        self.expand_input = nn.Sequential(nn.Linear(k_in, k_state), nn.Tanh())

        self.Lx = nn.Linear(k_state, 3 * k_state, bias=False)
        self.Lh_gate = nn.Linear(k_state, 2 * k_state, bias=True)
        self.Lh_lin = nn.Linear(k_state, k_state, bias=True)

        self.Ly = nn.Linear(k_state, k_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.init_params()
        self.train()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.k_state)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

        #initialize GRU reset gate bias to -1
        torch.nn.init.constant_(self.Lh_gate.bias[self.k_state:], -1)

    def forward(self, x, state, dt, x_half=None, x_full=None, **kwargs):
        if self.training:
            scheme = self.train_scheme
        else:
            scheme = self.train_scheme if self.eval_scheme == 'same' else self.eval_scheme

        state_new = self.dropout(RK(self.expand_input(x), state, self.f, dt, scheme, x_half=x_half, x_full=x_full))  # (batch, k_gru)
        #TODO check if this is best way to include input expansion
        y_new = self.Ly(state_new)
        return y_new, state_new

    def f(self, x, h):
        # x (batch, k_in), h (batch, k_gru)
        Lx = self.Lx(x)
        Lh_gate = self.Lh_gate(h)
        Lx_gate, Lx_lin = torch.split(Lx, [2 * self.k_state, self.k_state], dim=1)
        gates = torch.sigmoid(Lx_gate + Lh_gate)  # (batch, 2 * k_gru)
        z, r = torch.split(gates, [self.k_state, self.k_state], dim=1)

        return z * (torch.tanh(Lx_lin + self.Lh_lin(r * h)) - h) / self.meandt
        #return z * (self.dropout(torch.tanh(Lx_lin + self.Lh_lin(r * h))) - h)  # dropout in GRU hidden state update vector only


class IncrHOGRUCell(HOGRUCell):
    #potentially becomes unbounded!
    def f(self, x, h):
        # x (batch, k_in), h (batch, k_gru)
        Lx = self.Lx(x)
        Lh_gate = self.Lh_gate(h)
        Lx_gate, Lx_lin = torch.split(Lx, [2 * self.k_state, self.k_state], dim=1)
        gates = torch.sigmoid(Lx_gate + Lh_gate)  # (batch, 2 * k_gru)
        z, r = torch.split(gates, [self.k_state, self.k_state], dim=1)
        return z * (torch.tanh(Lx_lin + self.Lh_lin(r * h))) / self.meandt





class ASLinear(nn.Linear):
    """
    antisymmetric linear transform
    return: (W - W^T - gamma*E) x
    basic implementation (upper triangular would be more efficient)
    """
    def __init__(self, features, bias=True, gamma=0.01):
        super(ASLinear, self).__init__(features, features, bias) #set self.weight and self.bias
        self.register_buffer('diff', gamma * torch.eye(features)) #diffusion to avoid bad conditioning

    def forward(self, x):
        return nn.functional.linear(x, self.weight - self.weight.t() - self.diff, self.bias)





class HOARNNCell(nn.Module):
    """
    higher order Antisymmetric RNN cell;
    first-order scheme differs from ARNNCell in compensation with the state h in F!
    """
    def __init__(self, k_in, k_out, k_state, dropout=0., meandt=1, train_scheme='Euler', eval_scheme='Euler',
                 gamma=.01, step_size=1., **kwargs):

        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.meandt = meandt
        self.state_size = k_state
        self.train_scheme = train_scheme
        self.eval_scheme = eval_scheme
        self.step_size = step_size

        self.expand_input = nn.Sequential(nn.Linear(k_in, k_state), nn.Tanh())

        self.Lx = nn.Linear(k_state, 2 * k_state, bias=True)
        self.Lh = ASLinear(k_state, bias=False, gamma=gamma)  # same hidden-to-hidden for new state and gate

        self.Ly = nn.Linear(k_state, k_out, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.init_params()
        self.train()

    def init_params(self):
        stdv = 1.0 / math.sqrt(self.k_state)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, state, dt=None, x_half=None, x_full=None, **kwargs):
        if self.training:
            scheme = self.train_scheme
        else:
            scheme = self.train_scheme if self.eval_scheme == 'same' else self.eval_scheme

        if dt is None:
            dt = self.meandt  # no time info used
        state_new = self.dropout(RK(self.expand_input(x), state, self.f, dt, scheme, x_half=x_half, x_full=x_full))  # (batch, k_gru)

        y_new = self.Ly(state_new)
        return y_new, state_new

    def f(self, x, h):
        # x (batch, k_in), h (batch, k_state)
        Lx = self.Lx(x)  # standard linear transform
        Lh = self.Lh(h)  # anti-symmetric linear transform for hidden-to-hidden matrix
        Lx_gate, Lx_lin = torch.split(Lx, [self.k_state, self.k_state], dim=1)
        z = torch.sigmoid(Lx_gate + Lh)  # (batch, k_gru)

        return self.step_size * (z * torch.tanh(Lx_lin + Lh) - h) / self.meandt  # step_size constant pre-factor for function F (see ARNN paper)


class IncrHOARNNCell(HOARNNCell):
    """
    no compensation for linear term; 1st order scheme for equidistant data corresponds to original
    Antisymmetric RNN
    """

    def f(self, x, h):
        # x (batch, k_in), h (batch, k_state)
        Lx = self.Lx(x)  # standard linear transform
        Lh = self.Lh(h)  # anti-symmetric linear transform for hidden-to-hidden matrix
        Lx_gate, Lx_lin = torch.split(Lx, [self.k_state, self.k_state], dim=1)
        z = torch.sigmoid(Lx_gate + Lh)  # (batch, k_gru)

        return self.step_size * (z * torch.tanh(Lx_lin + Lh)) / self.meandt  # step_size constant pre-factor for function F (see ARNN paper)




class MIMO(nn.Module):

    def __init__(self, k_in, k_out, k_state, dropout=0., cell_factory=GRUCell, interpol="constant", **kwargs):
        # potential kwargs: meandt, train_scheme, eval_scheme

        super().__init__()
        self.k_in = k_in
        self.k_out = k_out
        self.k_state = k_state
        self.dropout = dropout
        self.criterion = nn.MSELoss()
        #self.criterion = nn.SmoothL1Loss()  # huber loss
        self.interpol = interpol

        self.cell = cell_factory(k_in, k_out, k_state, dropout=dropout, **kwargs)
        #self.register_buffer('state0', torch.zeros(k_state, ))  # fixed zero initial state
        self.state0 = nn.Parameter(torch.zeros(k_state,), requires_grad=True)  # trainable initial state


    def forward(self, inputs, state0=None, dt=None, **kwargs):  # potential kwargs: dt
        """
        inputs size (batch_size, seq_len, k_in)
        state0 is None (self.state0 used) or initial state with shape (batch_size, k_state)
        """
        state = self.state0.unsqueeze(0).expand(inputs.size(0), -1) if state0 is None else state0
        outputs = []
        states = []

        #interpolation of inputs for higher order models (quick and dirty rather than very general)
        if self.interpol == "constant":
            inputs_half = inputs
            inputs_full = inputs
        elif self.interpol == "linear":
            #TODO: extrapolation?  Just for comparison.
            inputs_last = inputs[:, -1, :].unsqueeze(1)
            inputs_half = 0.5 * (inputs[:, :-1, :] + inputs[:, 1:, :])
            inputs_half = torch.cat([inputs_half, inputs_last], dim=1)  # constant approximation at end of seq.
            inputs_full = inputs[:, 1:, :]
            inputs_full = torch.cat([inputs_full, inputs_last], dim=1)  # constant approximation at end of seq.
        elif self.interpol == 'predicted':
            raise NotImplementedError('Interpolation %s not yet implemented.'%str(self.interpol))
        else:
            raise NotImplementedError('invalid interpolation'%str(self.interpol))

        #forward pass through inputs sequence
        for i in range(inputs.size(1)):
            x0 = inputs[:, i, :]
            x_half = self.cell.expand_input(inputs_half[:, i, :])
            x_full = self.cell.expand_input(inputs_full[:, i, :])
            dt_i = None if dt is None else dt[:, i, :]
            output, state = self.cell(x0, state, dt=dt_i, x_half=x_half, x_full=x_full)  # output (batch, 2)
            outputs.append(output)
            states.append(state)

        outputs = torch.stack(outputs, dim=1)  # outputs: (batch, seq_len, 2)
        states = torch.stack(states, dim=1)

        return outputs, states


