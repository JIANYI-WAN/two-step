%% Clear environment variables
warning off             % Turn off warning messages
close all               % Close all open figure windows
clear                   % Clear variables
clc                     % Clear command window

%% Import data
load C:\Users\USER\Desktop\VibrationTable\Data\EI.mat
inputdata=[];

%% Data analysis
num_size = 0.7;                              % Proportion of the dataset for training
outdim = 1;                                  % The last column is the output
num_samples = size(res, 1);                  % Number of samples
res = res(randperm(num_samples), :);         % Shuffle the dataset (comment this line if shuffling is not desired)
num_train_s = round(num_size * num_samples); % Number of training samples
f_ = size(res, 2) - outdim;                  % Input feature dimension

%% Split the dataset into training and test sets
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%% Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%% Reshape data
% Reshaping the data into 1D is one way of processing
% It can also be reshaped into 2D or 3D depending on the model structure
% However, it should always be consistent with the input layer data structure
p_train =  double(reshape(p_train, f_, 1, 1, M));
p_test  =  double(reshape(p_test , f_, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';

%% Convert data format
for i = 1 : M
    Lp_train{i, 1} = p_train(:, :, 1, i);
end

for i = 1 : N
    Lp_test{i, 1}  = p_test( :, :, 1, i);
end

%% Build the model
lgraph = layerGraph();                                                 % Create an empty network structure

tempLayers = [
    sequenceInputLayer([f_, 1, 1], "Name", "sequence")                 % Input layer with data structure [f_, 1, 1]
    sequenceFoldingLayer("Name", "seqfold")];                          % Sequence folding layer
lgraph = addLayers(lgraph, tempLayers);                                % Add the above network structure to the empty structure

tempLayers = convolution2dLayer([3, 1], 32, "Name", "conv_1");         % Convolution layer with kernel [3, 1], stride [1, 1], 32 channels
lgraph = addLayers(lgraph, tempLayers);                                 % Add the above network structure to the empty structure

tempLayers = [
    reluLayer("Name", "relu_1")                                        % Activation layer
    convolution2dLayer([3, 1], 64, "Name", "conv_2")                   % Convolution layer with kernel [3, 1], stride [1, 1], 64 channels
    reluLayer("Name", "relu_2")];                                      % Activation layer
lgraph = addLayers(lgraph, tempLayers);                                % Add the above network structure to the empty structure

tempLayers = [
    globalAveragePooling2dLayer("Name", "gapool")                      % Global average pooling layer
    fullyConnectedLayer(16, "Name", "fc_2")                            % SE attention mechanism, 1/4 of the channel size
    reluLayer("Name", "relu_3")                                        % Activation layer
    fullyConnectedLayer(64, "Name", "fc_3")                            % SE attention mechanism, same number as the channels
    sigmoidLayer("Name", "sigmoid")];                                  % Activation layer
lgraph = addLayers(lgraph, tempLayers);                                % Add the above network structure to the empty structure

tempLayers = multiplicationLayer(2, "Name", "multiplication");         % Element-wise multiplication for attention
lgraph = addLayers(lgraph, tempLayers);                                % Add the above network structure to the empty structure

tempLayers = [
    sequenceUnfoldingLayer("Name", "sequnfold")                        % Sequence unfolding layer
    flattenLayer("Name", "flatten")                                    % Flatten layer
    gruLayer(6, "Name", "lstm", "OutputMode", "last")                  % LSTM layer
    fullyConnectedLayer(1, "Name", "fc")                               % Fully connected layer
    regressionLayer("Name", "regressionoutput")];                      % Regression output layer
lgraph = addLayers(lgraph, tempLayers);                                % Add the above network structure to the empty structure

lgraph = connectLayers(lgraph, "seqfold/out", "conv_1");               % Connect sequence folding layer output to convolution layer input
lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize"); 
                                                                       % Connect sequence folding output to sequence unfolding input  
lgraph = connectLayers(lgraph, "conv_1", "relu_1");                    % Connect convolution layer output to activation layer
lgraph = connectLayers(lgraph, "conv_1", "gapool");                    % Connect convolution layer output to global average pooling
lgraph = connectLayers(lgraph, "relu_2", "multiplication/in2");        % Connect activation layer output to multiplication layer
lgraph = connectLayers(lgraph, "sigmoid", "multiplication/in1");       % Connect fully connected output to multiplication layer
lgraph = connectLayers(lgraph, "multiplication", "sequnfold/in");      % Connect multiplication output to sequence unfolding input

%% Parameter settings
options = trainingOptions('adam', ...      % Adam optimizer
    'MaxEpochs', 1000, ...                 % Maximum number of iterations
    'InitialLearnRate', 1e-2, ...          % Initial learning rate of 0.01
    'LearnRateSchedule', 'piecewise', ...  % Learning rate decay
    'LearnRateDropFactor', 0.1, ...        % Learning rate decay factor of 0.1
    'LearnRateDropPeriod', 700, ...        % After 700 iterations, the learning rate is multiplied by 0.1
    'Shuffle', 'every-epoch', ...          % Shuffle the dataset every epoch
    'Plots', 'training-progress', ...      % Plot training progress
    'Verbose', false);

%% Train the model
net = trainNetwork(Lp_train, t_train, lgraph, options);

%% Model prediction
t_sim1 = predict(net, Lp_train);
t_sim2 = predict(net, Lp_test);

%% Reverse data normalization
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%% Root Mean Square Error
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

%% Display network structure
analyzeNetwork(net)

%% Plotting
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('True values', 'Predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
string = {'Training set prediction comparison'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('True values', 'Predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
string = {'Test set prediction comparison'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%% Calculate performance metrics
% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['R2 for training data: ', num2str(R1)])
disp(['R2 for test data: ', num2str(R2)])

% MAE
mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

disp(['MAE for training data: ', num2str(mae1)])
disp(['MAE for test data: ', num2str(mae2)])

% MBE
mbe1 = sum(T_sim1' - T_train) ./ M ;
mbe2 = sum(T_sim2' - T_test ) ./ N ;

disp(['MBE for training data: ', num2str(mbe1)])
disp(['MBE for test data: ', num2str(mbe2)])
