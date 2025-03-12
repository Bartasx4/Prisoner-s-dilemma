import torch
import torch.nn as nn


Activation = torch._C._VariableFunctions


class PolicyNetwork(nn.Module):
    # ReLU activation logic

    def __init__(self, hidden_size: int, output_activation: Activation, hidden_activation: Activation):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.output_activation: Activation = output_activation
        self.hidden_activation: Activation = hidden_activation

    def forward(self, state):
        pass

    @property
    def info(self) -> list[tuple[str, str | int]]:
        return [
            ('name', self.__module__),
            ('output_activation', self.output_activation.__name__),
            ('hidden_activation', self.hidden_activation.__name__),
            ('hidden_size', self.hidden_size),
        ]


class LSTMSoftmaxReluPolicyNetwork(PolicyNetwork):
    OUTPUT_ACTIVATION = torch.softmax
    HIDDEN_ACTIVATION = torch.relu

    def __init__(self, hidden_size:int = 16):
        super().__init__(hidden_size, self.OUTPUT_ACTIVATION, self.HIDDEN_ACTIVATION)
        self.lstm_layer = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        lstm_output, _ = self.lstm_layer(inputs)
        hidden_output = self.relu_activation(lstm_output[:, -1, :])
        return self.output_activation(self.output_layer(hidden_output), dim=-1)

    def relu_activation(self, tensor):
        return self.hidden_activation(tensor)


class LSTMSigmoidPolicyNetwork(PolicyNetwork):
    OUTPUT_ACTIVATION = torch.sigmoid
    HIDDEN_ACTIVATION = torch.sigmoid

    def __init__(self, hidden_size:int = 16):
        super().__init__(hidden_size, self.OUTPUT_ACTIVATION, self.HIDDEN_ACTIVATION)
        self.lstm_layer = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        lstm_output, _ = self.lstm_layer(inputs)
        hidden_output = self.relu_activation(lstm_output[:, -1, :])
        return self.output_activation(self.output_layer(hidden_output), dim=-1)

    def relu_activation(self, tensor):
        return self.hidden_activation(tensor)
