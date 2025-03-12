import torch
import torch.nn as nn


Activation = torch._C._VariableFunctions


class PolicyNetwork(nn.Module):
    """Base class for a policy network. Defines the structure and activation functions."""

    def __init__(self, hidden_size: int, output_activation: Activation, hidden_activation: Activation):
        """
        Initialize the PolicyNetwork with hidden and output activation functions.
        """
        super().__init__()
        self.hidden_size: int = hidden_size
        self.output_activation: Activation = output_activation
        self.hidden_activation: Activation = hidden_activation

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for the policy network. To be implemented by subclasses."""
        raise NotImplementedError("the forward method must be implemented in the subclass.")

    @property
    def info(self) -> list[tuple[str, str | int]]:
        """Returns the policy network's configuration details."""
        return [
            ('name', self.__module__),
            ('output_activation', self.output_activation.__name__),
            ('hidden_activation', self.hidden_activation.__name__),
            ('hidden_size', self.hidden_size),
        ]


class LSTMSoftmaxReluPolicyNetwork(PolicyNetwork):
    """Policy network with LSTM layers, ReLU for hidden layers, and Softmax for output layers."""

    OUTPUT_ACTIVATION = torch.softmax
    HIDDEN_ACTIVATION = torch.relu

    def __init__(self, hidden_size: int = 16):
        """
        Initializes the LSTM + ReLU + Softmax policy network.
        """
        super().__init__(hidden_size, self.OUTPUT_ACTIVATION, self.HIDDEN_ACTIVATION)
        self.lstm_layer = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, inputs):
        lstm_output, _ = self.lstm_layer(inputs)
        hidden_output = self.hidden_activation(lstm_output[:, -1, :])
        return self.output_activation(self.output_layer(hidden_output), dim=-1)


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
        hidden_output = self.hidden_activation(lstm_output[:, -1, :])
        return self.output_activation(self.output_layer(hidden_output), dim=-1)
