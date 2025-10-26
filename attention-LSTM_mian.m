%% Clear environment variables
warning off             % Turn off warning messages
close all               % Close all open figure windows
clear                   % Clear variables
clc                     % Clear command window

%% Import data
load E:\Project\Data\'data EI 0.12.mat'
result = alldata(:,2:3);

%% Data analysis
num_samples = length(result);  % Number of samples
kim = 1;                      % Delay step (previous history data as independent variable)
zim =  1;                      % Predict across zim time steps
nim = size(result, 2) - 1;     % Number of features in original data

%% Split the dataset
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(result(i: i + kim - 1 + zim, 1: end - 1)', 1, ...
        (kim + zim) * nim), result(i + kim + zim - 1, end)];
end

%% Dataset analysis
outdim = 1;                                  % The last column is the output
num_size = 0.7;                              % Proportion of training data
num_train_s = round(num_size * num_samples); % Number of training samples
f_ = size(res, 2) - outdim;                  % Input feature length
num_val = 0.2;                               % Validation set proportion

%% Split into training and test sets
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%% Data normalisation
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%% Data reshaping
p_train = reshape(p_train, nim, f_ / nim, M);
p_test  = reshape(p_test , nim, f_ / nim, N);
p_train = permute(p_train, [1, 3, 2]);
p_test  = permute(p_test , [1, 3, 2]);
t_train = t_train';
t_test  = t_test' ;

%% Format conversion
p_train = dlarray(p_train, 'CBT');
t_train = dlarray(t_train, 'BC' );
p_test  = dlarray(p_test , 'CBT');
t_test  = dlarray(t_test , 'BC' );

%% Split validation set
val_size = floor(M * num_val);

p_vrain = p_train(:, M - val_size + 1: M, :);
t_vrain = t_train(:, M - val_size + 1: M);

p_train = p_train(:, 1: M - val_size, :);
t_train = t_train(:, 1: M - val_size);

%% Parameter settings
indim = size(p_train, 1);    % Data feature dimension
num_hidden = 50;             % Number of hidden layer nodes

%% Parameter initialisation
[params, State] = paramsInit(num_hidden, indim, outdim);

%% Training parameter settings
Epochs = 50;                                              % Maximum number of iterations
minibatch = 32;                                           % Training batch size
LearnRateSchedule='piecewise';
LearnRateDropPeriod=125;
LearnRateDropFactor=0.2;                                  % Learning rate drop factor
LearnRate = 0.00001;                                       % Initial learning rate
validationFrequency = 10;                                 % Cross-validation frequency
IterationsPerEpoch = floor((M - val_size) / minibatch);   % Number of iterations per epoch

%% Gradient update parameters
averageGrad = [];
averageSqGrad = [];

%% Plot loss curve
figure
start = tic;
LossTrain = animatedline('color', 'r', 'LineWidth', 1);
LossValid = animatedline('color', 'b', 'LineWidth', 1);
xlabel("Iteration")
ylabel("Loss")

%% Model training
iteration = 0;
for epoch = 1: Epochs
    epoch

    % Initialising model parameters for each epoch
    [~, state] = paramsInit(num_hidden, indim, outdim);       

    % Updating training for each batch per epoch
    for i = 1 : IterationsPerEpoch

        % Record iteration number
        iteration = iteration + 1;
        idx = (i - 1) * minibatch + 1: i * minibatch;
        
        % Get input and output for this batch
        dlX = gpuArray(p_train(:, idx, :));
        dlY = gpuArray(t_train(idx));

        % Train the model and get updated gradients
        [gradients, loss, state] = dlfeval(@Model, dlX, dlY, params, state);

        % Optimise using Adam
        [params, averageGrad, averageSqGrad] = adamupdate(params, gradients, averageGrad, averageSqGrad, iteration, LearnRate);
        
        % Validate using validation set
        if iteration == 1 || mod(iteration, validationFrequency) == 0
            output_Ynorm = ModelPredict(gpuArray(p_vrain), params, State);
            lossValidation = mse(output_Ynorm, gpuArray(t_vrain));
        end
        
        % Plot training loss curve
        D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
        addpoints(LossTrain, iteration, double(gather(extractdata(loss))))

        % Plot validation loss curve
        if iteration == 1 || mod(iteration,validationFrequency) == 0
            addpoints(LossValid, iteration, double(gather(extractdata(lossValidation))))
        end
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        legend('Training', 'Validation')
        drawnow
        
    end
    
    % Update learning rate
    if mod(epoch, 5) == 0
        LearnRate = LearnRate * LearnRateDropFactor;
    end

end

%% Simulation testing
t_sim1 = ModelPredict(gpuArray(p_train), params, State);
t_sim2 = ModelPredict(gpuArray(p_vrain), params, State);
t_sim3 = ModelPredict(gpuArray(p_test) , params, State);

%% Extract data
t_sim1 = extractdata(t_sim1);
t_sim2 = extractdata(t_sim2);
t_sim3 = extractdata(t_sim3);

%% Reverse normalisation
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_sim3 = mapminmax('reverse', t_sim3, ps_output);

%% Split validation set
T_vrain = T_train(:, M - val_size + 1: M);
T_train = T_train(:, 1: M - val_size);

%% Root Mean Square Error
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ length(T_sim1));
error2 = sqrt(sum((T_sim2 - T_vrain).^2) ./ length(T_sim2));
error3 = sqrt(sum((T_sim3 - T_test ).^2) ./ length(T_sim3));

%% Plot
figure
plot(1: length(T_sim1), T_train, 'r-*', 1: length(T_sim1), T_sim1, 'b-o', 'LineWidth', 1)
legend('True values', 'attention-LSTM predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
string = {'Training set prediction comparison'; ['RMSE=' num2str(error1)]};
title(string)
xlim([1, length(T_sim1)])
grid

figure
plot(1: length(T_sim2), T_vrain, 'r-*', 1: length(T_sim2), T_sim2, 'b-o', 'LineWidth', 1)
legend('True values', 'attention-LSTM predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
string = {'Validation set prediction comparison'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, length(T_sim2)])
grid

figure
plot(1: length(T_sim3), T_test, 'r-*', 1: length(T_sim3), T_sim3, 'b-o', 'LineWidth', 1)
legend('True values', 'attention-LSTM predicted values')
xlabel('Prediction samples')
ylabel('Prediction results')
string = {'Test set prediction comparison'; ['RMSE=' num2str(error3)]};
title(string)
xlim([1, length(T_sim3)])
grid

%% Test set error plot
figure  
ERROR3=T_test-T_sim3;
plot(T_test-T_sim3,'b-*','LineWidth',1.5)
xlabel('Test sample number')
ylabel('Prediction error')
title('Test set prediction error')
grid on;
legend('attention-LSTM prediction error')

%% Performance metrics calculation
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_vrain - T_sim2)^2 / norm(T_vrain - mean(T_vrain))^2;
R3 = 1 - norm(T_test  - T_sim3)^2 / norm(T_test  - mean(T_test ))^2;
 
disp(['R2 for training data: ', num2str(R1)])
disp(['R2 for validation data: ', num2str(R2)])
disp(['R2 for test data: ', num2str(R3)])

% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ length(T_sim1) ;
mae2 = sum(abs(T_sim2 - T_vrain)) ./ length(T_sim2) ;
mae3 = sum(abs(T_sim3 - T_test )) ./ length(T_sim3) ;

disp(['MAE for training data: ', num2str(mae1)])
disp(['MAE for validation data: ', num2str(mae2)])
disp(['MAE for test data: ', num2str(mae3)])

% MBE
mbe1 = sum(T_sim1 - T_train) ./ length(T_sim1) ;
mbe2 = sum(T_sim2 - T_vrain) ./ length(T_sim2) ;
mbe3 = sum(T_sim3 - T_test ) ./ length(T_sim3) ;

disp(['MBE for training data: ', num2str(mbe1)])
disp(['MBE for validation data: ', num2str(mbe2)])
disp(['MBE for test data: ', num2str(mbe3)])

figure
plot(T_test,T_sim3,'ob');
xlabel('True values')
ylabel('Predicted values')
string1 = {'Test set performance';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
