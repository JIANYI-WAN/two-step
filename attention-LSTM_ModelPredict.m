function t_sim = ModelPredict(p_test, params, state)

%% Get LSTM network weights
lstmbias = params.lstm.bias;
lstmweight = params.lstm.weights;
lstmrecurrentWeights = params.lstm.recurrentWeights;
num_hidden = size(lstmrecurrentWeights, 2);

%% In each epoch, the state is passed between different batches but not learned
h0 = state.lstm.h0;
c0 = state.lstm.c0;
Lstm_Y = lstm(p_test, h0, c0, lstmweight, lstmrecurrentWeights, lstmbias);

%% Attention parameter setup
Attentionweight = params.attention.weight;    % Score weight
Ht = Lstm_Y(:, :, end);                       % Reference vector
num_time = size(Lstm_Y, 3);                   % Time scale

%% Attention scores
socre = dlarray;
for i = 1: num_time - 1
    A = extractdata(squeeze(Lstm_Y(:, :, i)));
    A = repmat(A, [1, 1, num_hidden]);
    A = permute(A, [1, 3, 2]);
    A = dlarray(A, 'SCB');
    B = squeeze(sum(A .* dlarray(Attentionweight, 'SC'), 1));
    C = squeeze(sum(B .* Ht, 1));
    socre = [socre; C];
end

%% Compute scores
a = sigmoid(socre);
Vt = 0;
for i = 1: num_time - 1
    Vt = Vt + a(i, :) .* Lstm_Y(:, :, i);
end

%% Attention mechanism output
bias1 = params.attenout.bias1;
bias2 = params.attenout.bias2;
weight1 = params.attenout.weight1;
weight2 = params.attenout.weight2;

HVT = fullyconnect(Vt, weight1, bias1) + fullyconnect(Ht, weight2, bias2);

%% Fully connected layer
LastBias = params.fullyconnect.bias1;
LastWeight = params.fullyconnect.weight1;

FCI = fullyconnect(HVT, LastWeight, LastBias);
FCI = relu(FCI);

%% Regression layer weights
fullybias = params.fullyconnect.bias2;
fullyweight = params.fullyconnect.weight2;

t_sim = fullyconnect(FCI, fullyweight, fullybias);
t_sim = squeeze(t_sim);
% t_sim = relu(t_sim);

end
