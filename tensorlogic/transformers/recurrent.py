"""
Recurrent Neural Networks (RNN, LSTM, GRU) implemented as tensor equations.

In Tensor Logic, recurrent computations are tensor equations with temporal indices:
- h[t] := f(x[t] × W_x + h[t-1] × W_h + b)
- LSTM gates are tensor equations with sigmoid/tanh activations
- Everything reduces to einsum operations + nonlinearities

This module provides:
- SimpleRNN: Basic recurrent layer
- LSTM: Long Short-Term Memory
- GRU: Gated Recurrent Unit
- BidirectionalWrapper: Makes any RNN bidirectional
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Any


class TensorEquationRNN(nn.Module):
    """
    Base class for RNNs implemented as tensor equations.

    Core idea: Recurrent computation is a temporal tensor equation:
    h[b,t,d] := activation(x[b,t,e] × W_x[e,d] + h[b,t-1,d] × W_h[d,d] + b[d])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        mode: str = 'continuous',
        temperature: float = 1.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.temperature = temperature

    def tensor_equation_step(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        W_x: torch.Tensor,
        W_h: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        activation: str = 'tanh'
    ) -> torch.Tensor:
        """
        Single step of RNN as tensor equation.

        Equation: h_t = activation(x_t × W_x + h_prev × W_h + b)

        Args:
            x_t: [batch_size, input_size] input at time t
            h_prev: [batch_size, hidden_size] hidden state at t-1
            W_x: [input_size, hidden_size] input weight matrix
            W_h: [hidden_size, hidden_size] hidden weight matrix
            bias: [hidden_size] optional bias
            activation: 'tanh', 'relu', 'sigmoid', or 'none'

        Returns:
            h_t: [batch_size, hidden_size] new hidden state
        """
        # Input transformation: x_t × W_x using einsum
        # bi,ih->bh (batch×input, input×hidden -> batch×hidden)
        input_contrib = torch.einsum('bi,ih->bh', x_t, W_x)

        # Hidden transformation: h_prev × W_h using einsum
        # bh,hd->bd (batch×hidden, hidden×hidden -> batch×hidden)
        hidden_contrib = torch.einsum('bh,hd->bd', h_prev, W_h)

        # Combine with bias
        pre_activation = input_contrib + hidden_contrib
        if bias is not None:
            pre_activation = pre_activation + bias

        # Apply activation
        if activation == 'tanh':
            h_t = torch.tanh(pre_activation)
        elif activation == 'relu':
            h_t = F.relu(pre_activation)
        elif activation == 'sigmoid':
            h_t = torch.sigmoid(pre_activation)
        elif activation == 'none':
            h_t = pre_activation
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # In boolean mode, threshold to binary
        if self.mode == 'boolean':
            h_t = (h_t > 0.5).float()

        return h_t


class SimpleRNN(TensorEquationRNN):
    """
    Simple RNN layer using tensor equations.

    Tensor equation:
    h[t] = tanh(x[t] × W_x + h[t-1] × W_h + b)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        mode: str = 'continuous',
        activation: str = 'tanh'
    ):
        super().__init__(input_size, hidden_size, bias, mode)
        self.activation = activation

        # Weight matrices as learnable parameters
        self.W_x = nn.Parameter(torch.randn(input_size, hidden_size) / (input_size ** 0.5))
        self.W_h = nn.Parameter(torch.randn(hidden_size, hidden_size) / (hidden_size ** 0.5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias', None)

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        return_sequences: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through RNN.

        Args:
            x: [batch_size, seq_len, input_size] input tensor
            h_0: [batch_size, hidden_size] initial hidden state
            return_sequences: If True, return all hidden states; else return only last

        Returns:
            If return_sequences:
                output: [batch_size, seq_len, hidden_size] all hidden states
                h_n: [batch_size, hidden_size] final hidden state
            Else:
                h_n: [batch_size, hidden_size] final hidden state only
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden state
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        # Process sequence
        h_t = h_0
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_size]
            h_t = self.tensor_equation_step(
                x_t, h_t, self.W_x, self.W_h, self.bias, self.activation
            )
            outputs.append(h_t)

        # Stack outputs
        if return_sequences:
            output = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
            return output, h_t
        else:
            return h_t


class LSTM(TensorEquationRNN):
    """
    LSTM layer using tensor equations.

    Tensor equations for LSTM gates:
    i[t] = sigmoid(x[t] × W_xi + h[t-1] × W_hi + b_i)  # Input gate
    f[t] = sigmoid(x[t] × W_xf + h[t-1] × W_hf + b_f)  # Forget gate
    g[t] = tanh(x[t] × W_xg + h[t-1] × W_hg + b_g)     # Candidate values
    o[t] = sigmoid(x[t] × W_xo + h[t-1] × W_ho + b_o)  # Output gate

    Cell state update:
    c[t] = f[t] ⊙ c[t-1] + i[t] ⊙ g[t]
    h[t] = o[t] ⊙ tanh(c[t])

    Where ⊙ denotes element-wise multiplication.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        mode: str = 'continuous'
    ):
        super().__init__(input_size, hidden_size, bias, mode)

        # Weight matrices for all gates (i, f, g, o)
        self.W_x = nn.Parameter(torch.randn(4, input_size, hidden_size) / (input_size ** 0.5))
        self.W_h = nn.Parameter(torch.randn(4, hidden_size, hidden_size) / (hidden_size ** 0.5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(4, hidden_size))
        else:
            self.register_parameter('bias', None)

    def lstm_cell(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single LSTM cell computation as tensor equations.

        Args:
            x_t: [batch_size, input_size] input at time t
            h_prev: [batch_size, hidden_size] previous hidden state
            c_prev: [batch_size, hidden_size] previous cell state

        Returns:
            h_t: [batch_size, hidden_size] new hidden state
            c_t: [batch_size, hidden_size] new cell state
        """
        # Compute all gates in parallel using batched einsum
        # gates shape: [4, batch_size, hidden_size]
        gates_x = torch.einsum('gih,bi->gbh', self.W_x, x_t)
        gates_h = torch.einsum('ghd,bh->gbd', self.W_h, h_prev)

        gates = gates_x + gates_h
        if self.bias is not None:
            gates = gates + self.bias.unsqueeze(1)

        # Split and apply activations
        i_t = torch.sigmoid(gates[0])  # Input gate
        f_t = torch.sigmoid(gates[1])  # Forget gate
        g_t = torch.tanh(gates[2])     # Candidate values
        o_t = torch.sigmoid(gates[3])  # Output gate

        # Update cell state: c[t] = f[t] ⊙ c[t-1] + i[t] ⊙ g[t]
        c_t = f_t * c_prev + i_t * g_t

        # Compute hidden state: h[t] = o[t] ⊙ tanh(c[t])
        h_t = o_t * torch.tanh(c_t)

        # Boolean mode: threshold states
        if self.mode == 'boolean':
            h_t = (h_t > 0.5).float()
            c_t = (c_t > 0.5).float()

        return h_t, c_t

    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequences: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through LSTM.

        Args:
            x: [batch_size, seq_len, input_size] input tensor
            initial_states: Optional (h_0, c_0) tuple
            return_sequences: If True, return all hidden states

        Returns:
            If return_sequences:
                output: [batch_size, seq_len, hidden_size]
                (h_n, c_n): Final states
            Else:
                (h_n, c_n): Final states only
        """
        batch_size, seq_len, _ = x.shape

        # Initialize states
        if initial_states is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h_0, c_0 = initial_states

        # Process sequence
        h_t, c_t = h_0, c_0
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
            outputs.append(h_t)

        # Return results
        if return_sequences:
            output = torch.stack(outputs, dim=1)
            return output, (h_t, c_t)
        else:
            return (h_t, c_t)


class GRU(TensorEquationRNN):
    """
    GRU layer using tensor equations.

    Tensor equations for GRU:
    z[t] = sigmoid(x[t] × W_xz + h[t-1] × W_hz + b_z)  # Update gate
    r[t] = sigmoid(x[t] × W_xr + h[t-1] × W_hr + b_r)  # Reset gate
    h̃[t] = tanh(x[t] × W_xh + (r[t] ⊙ h[t-1]) × W_hh + b_h)  # Candidate
    h[t] = (1 - z[t]) ⊙ h[t-1] + z[t] ⊙ h̃[t]  # New hidden state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        mode: str = 'continuous'
    ):
        super().__init__(input_size, hidden_size, bias, mode)

        # Weights for update gate, reset gate, and candidate
        self.W_x = nn.Parameter(torch.randn(3, input_size, hidden_size) / (input_size ** 0.5))
        self.W_h = nn.Parameter(torch.randn(3, hidden_size, hidden_size) / (hidden_size ** 0.5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(3, hidden_size))
        else:
            self.register_parameter('bias', None)

    def gru_cell(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Single GRU cell computation.

        Args:
            x_t: [batch_size, input_size]
            h_prev: [batch_size, hidden_size]

        Returns:
            h_t: [batch_size, hidden_size] new hidden state
        """
        # Compute gates using einsum
        gates_x = torch.einsum('gih,bi->gbh', self.W_x[:2], x_t)
        gates_h = torch.einsum('ghd,bh->gbd', self.W_h[:2], h_prev)

        gates = gates_x + gates_h
        if self.bias is not None:
            gates = gates + self.bias[:2].unsqueeze(1)

        # Update and reset gates
        z_t = torch.sigmoid(gates[0])  # Update gate
        r_t = torch.sigmoid(gates[1])  # Reset gate

        # Candidate hidden state
        h_cand_x = torch.einsum('ih,bi->bh', self.W_x[2], x_t)
        h_cand_h = torch.einsum('hd,bh->bd', self.W_h[2], r_t * h_prev)

        h_cand = h_cand_x + h_cand_h
        if self.bias is not None:
            h_cand = h_cand + self.bias[2]

        h_tilde = torch.tanh(h_cand)

        # New hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        # Boolean mode
        if self.mode == 'boolean':
            h_t = (h_t > 0.5).float()

        return h_t

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        return_sequences: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through GRU."""
        batch_size, seq_len, _ = x.shape

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h_t = h_0
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.gru_cell(x_t, h_t)
            outputs.append(h_t)

        if return_sequences:
            output = torch.stack(outputs, dim=1)
            return output, h_t
        else:
            return h_t


class BidirectionalWrapper(nn.Module):
    """
    Makes any RNN bidirectional.

    Processes sequence in both directions and concatenates results.
    """

    def __init__(self, rnn_module: nn.Module, merge_mode: str = 'concat'):
        """
        Args:
            rnn_module: An RNN module (SimpleRNN, LSTM, or GRU)
            merge_mode: How to combine forward/backward ('concat', 'sum', 'mul', 'avg')
        """
        super().__init__()
        self.forward_rnn = rnn_module

        # Create backward RNN with same architecture
        rnn_class = type(rnn_module)
        self.backward_rnn = rnn_class(
            input_size=rnn_module.input_size,
            hidden_size=rnn_module.hidden_size,
            bias=rnn_module.bias is not None
        )

        self.merge_mode = merge_mode

    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[Any] = None
    ) -> torch.Tensor:
        """
        Bidirectional forward pass.

        Args:
            x: [batch_size, seq_len, input_size]
            initial_states: Optional initial states

        Returns:
            output: [batch_size, seq_len, hidden_size * 2] if merge_mode='concat'
                   [batch_size, seq_len, hidden_size] otherwise
        """
        # Forward direction
        forward_out, _ = self.forward_rnn(x, initial_states, return_sequences=True)

        # Backward direction (reverse sequence)
        x_reversed = torch.flip(x, dims=[1])
        backward_out, _ = self.backward_rnn(x_reversed, initial_states, return_sequences=True)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back

        # Merge outputs
        if self.merge_mode == 'concat':
            output = torch.cat([forward_out, backward_out], dim=-1)
        elif self.merge_mode == 'sum':
            output = forward_out + backward_out
        elif self.merge_mode == 'mul':
            output = forward_out * backward_out
        elif self.merge_mode == 'avg':
            output = (forward_out + backward_out) / 2
        else:
            raise ValueError(f"Unknown merge mode: {self.merge_mode}")

        return output


def export_rnn_as_equations(rnn: TensorEquationRNN) -> List[str]:
    """
    Export RNN as tensor equations in symbolic form.

    Args:
        rnn: An RNN module

    Returns:
        List of equation strings
    """
    equations = []

    if isinstance(rnn, SimpleRNN):
        equations.append(f"# Simple RNN with {rnn.hidden_size} hidden units")
        equations.append(f"h[t] := {rnn.activation}(x[t] × W_x[{rnn.input_size},{rnn.hidden_size}] + "
                        f"h[t-1] × W_h[{rnn.hidden_size},{rnn.hidden_size}] + b[{rnn.hidden_size}])")

    elif isinstance(rnn, LSTM):
        equations.append(f"# LSTM with {rnn.hidden_size} hidden units")
        equations.append(f"i[t] := sigmoid(x[t] × W_xi + h[t-1] × W_hi + b_i)")
        equations.append(f"f[t] := sigmoid(x[t] × W_xf + h[t-1] × W_hf + b_f)")
        equations.append(f"g[t] := tanh(x[t] × W_xg + h[t-1] × W_hg + b_g)")
        equations.append(f"o[t] := sigmoid(x[t] × W_xo + h[t-1] × W_ho + b_o)")
        equations.append(f"c[t] := f[t] ⊙ c[t-1] + i[t] ⊙ g[t]")
        equations.append(f"h[t] := o[t] ⊙ tanh(c[t])")

    elif isinstance(rnn, GRU):
        equations.append(f"# GRU with {rnn.hidden_size} hidden units")
        equations.append(f"z[t] := sigmoid(x[t] × W_xz + h[t-1] × W_hz + b_z)")
        equations.append(f"r[t] := sigmoid(x[t] × W_xr + h[t-1] × W_hr + b_r)")
        equations.append(f"h̃[t] := tanh(x[t] × W_xh + (r[t] ⊙ h[t-1]) × W_hh + b_h)")
        equations.append(f"h[t] := (1 - z[t]) ⊙ h[t-1] + z[t] ⊙ h̃[t]")

    return equations