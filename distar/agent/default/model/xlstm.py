import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from collections import namedtuple
from typing import List, Tuple
from torch import Tensor
import numbers
import warnings

warnings.filterwarnings("ignore", message=r"'layers' was found in ScriptModule constants, but it is a non-constant submodule. Consider removing it.")

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

# xLSTM architecture components
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))
        self.normalizer_state = Parameter(torch.ones(hidden_size))

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # Normalize memory updates with a normalizer state
        cy = (forgetgate * cx) + (ingate * cellgate)
        normalized_cy = cy / self.normalizer_state
        hy = outgate * torch.tanh(normalized_cy)

        return hy, (hy, normalized_cy)


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, hidden_size))
        self.key_weight = Parameter(torch.randn(hidden_size, input_size))
        self.value_weight = Parameter(torch.randn(hidden_size, input_size))
        self.cov_matrix = Parameter(torch.zeros(hidden_size, hidden_size))

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        key = torch.mm(input, self.key_weight.t())
        value = torch.mm(input, self.value_weight.t())

        # Update covariance matrix
        self.cov_matrix = self.cov_matrix + torch.mm(value.t(), key)

        output = torch.mm(hx, self.weight_hh.t()) + torch.mm(input, self.weight_ih.t())
        hy = torch.tanh(output)
        return hy, (hy, cx)


class xLSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(xLSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs.append(out)
        return torch.stack(outputs), state


class StackedxLSTM(nn.Module):
    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedxLSTM, self).__init__()
        self.layers = [layer(*first_layer_args)] + [layer(*other_layer_args) for _ in range(num_layers - 1)]

    def forward(self, input: Tensor, states: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        output_states = []
        output = input
        for i, layer in enumerate(self.layers):
            state = states[i]
            output, out_state = layer(output, state)
            output_states.append(out_state)
        return output, output_states


def script_xlstm(input_size, hidden_size, num_layers, use_mLSTM=False):
    '''Returns a ScriptModule using xLSTM architecture.'''
    if use_mLSTM:
        return StackedxLSTM(num_layers, xLSTMLayer, [mLSTMCell, input_size, hidden_size], [mLSTMCell, hidden_size, hidden_size])
    else:
        return StackedxLSTM(num_layers, xLSTMLayer, [sLSTMCell, input_size, hidden_size], [sLSTMCell, hidden_size, hidden_size])


# Test function for xLSTM
def test_script_xlstm(seq_len, batch, input_size, hidden_size, num_layers, use_mLSTM=False):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size), torch.randn(batch, hidden_size)) for _ in range(num_layers)]
    rnn = script_xlstm(input_size, hidden_size, num_layers, use_mLSTM)

    # Run xLSTM architecture test
    out, out_state = rnn(inp, states)
    return out, out_state

