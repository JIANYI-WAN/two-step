function [params, state] = paramsInit(num_hidden, indim, outdim)

%% Initialise LSTM state
state.lstm.h0 =  gpuArray(dlarray(zeros(num_hidden, 1)));  % Initial hidden state (h0) for LSTM
state.lstm.c0 =  gpuArray(dlarray(zeros(num_hidden, 1)));  % Initial cell state (c0) for LSTM

%% Initialise LSTM parameters
params.lstm.bias = gpuArray(dlarray(0.01 * randn(4 * num_hidden, 1), 'C'));   % Biases for LSTM
params.lstm.weights  = gpuArray(dlarray(0.01 * randn(4 * num_hidden, indim), 'CU'));   % Weights for input-to-hidden connection in LSTM
params.lstm.recurrentWeights = gpuArray(dlarray(0.01 * randn(4 * num_hidden, num_hidden), 'CU'));   % Weights for recurrent connection in LSTM

%% Initialise attention mechanism parameters
params.attention.weight = gpuArray(dlarray(0.01 * randn(num_hidden, num_hidden)));  % Weights for attention score calculation

%% Initialise attention output parameters
params.attenout.weight1 = gpuArray(dlarray(0.01 * randn(num_hidden, num_hidden)));  % Weights for first attention output layer
params.attenout.weight2 = gpuArray(dlarray(0.01 * randn(num_hidden, num_hidden)));  % Weights for second attention output layer

params.attenout.bias1 = gpuArray(dlarray(0.01 * randn(num_hidden, 1)));  % Biases for first attention output layer
params.attenout.bias2 = gpuArray(dlarray(0.01 * randn(num_hidden, 1)));  % Biases for second attention output layer

%% Initialise fully connected layer parameters
params.fullyconnect.weight1 = gpuArray(dlarray(0.01 * randn(10, num_hidden)));   % Weights for the first fully connected layer
params.fullyconnect.weight2 = gpuArray(dlarray(0.01 * randn(outdim, 10)));   % Weights for the second fully connected layer

params.fullyconnect.bias1 = gpuArray(dlarray(0.01 * randn(10, 1)));   % Biases for the first fully connected layer
params.fullyconnect.bias2 = gpuArray(dlarray(0.01 * randn(outdim, 1)));   % Biases for the second fully connected layer

end
