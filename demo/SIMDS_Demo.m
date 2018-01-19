%% importanPattern.m
%
% Script that demponstartes the use of the SIMDS class. It tries to embed
% Grid2D Data, while idntifying the important samples. 
% This dataset contains a 10-dimensional, numeric samples from the 
% synthetic Grid dataset into a 2-dimensional embedding space.
% The SIMDS model is trained using an Iterative Majorization algorithm.
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%


%% MATLAB cleanup
clc; clear; close all

for index=1:5
    %% Load & prepare data
    load('C:\Users\Mahlagha\Documents\MATLAB\Data\grid2D_data') % loads train and test matries
    N = size(Xtrain,1);
    Nt = size(Xtest,1);
    train_idx = (1:1:N);
    
    
    % Compute matrix of pair-wise dissimilarities based on Euclidean distances
    Delta = SIMDS.calcEuclideanDistanceMatrix(Xtrain);
    Delta = Delta + min(Delta(Delta>0))*0.001;
    
    %% set up all kernels
    
    % Here we use a Guassian kernel with parameter sigmaSquared parameter
    sigmaSquaredVector = [2.5 ] .^ 2;
    data = [Xtrain;Xtest];
    Ktensor_all = zeros(N+Nt,N+Nt,length(sigmaSquaredVector));
    for s = 1 : length(sigmaSquaredVector)
        sigmaSquared = sigmaSquaredVector(s);
        Ktensor_all(:,:,s) = gaussianKernelMatrix(data,data,sigmaSquared);
    end
    Ktensor_test_valid = Ktensor_all(N+1:Nt+N,1:Nt,:);
    
    %% generate validation set
    valid_idx = randi(Nt,1,floor(0.3*Nt));
    Xvalid = Xtest(valid_idx,:);
    Nv = size(Xvalid,1);
    % Xtest again
    test_idx = (1:1:Nt);
    test_idx = setxor(test_idx,valid_idx);
    Xtest = Xtest(test_idx,:);
    Nt = size(Xtest,1);
    
    %% Setup Kernel tensor for training
    Ktensor = Ktensor_all(1:N,1:N,:);
    
    %% Setup Kernel tensor for testing
    Ktensor_test = Ktensor_test_valid(test_idx,1:N,:);
    
    
    %% Setup Kernel tensor for validation
    Ktensor_valid = Ktensor_test_valid(valid_idx,1:N,:);
    
    %% Specify training parameters
    
    % specify embedding dimension; we choose 2,
    % so we can visualize the results
    P = 2;
    
    % set U matrix so that all discrepancies are
    % considered equally important.
    U = ones(N,N) - eye(N);
    
    % To make the delta(m,m) = 0
    Delta = Delta.*U;
    
    
    %% Initialize weight matrix C and MKL vector Theta
    
    % fix theta to 1, since there is only one kernel
    Theta_init = 1;
    % randomize initial C
    C_init = randn(P,N);
    
    %% Create SIMDS object
    model = SIMDS(Ktensor, C_init, Theta_init, Delta, U);
    
    %% Train SIMDS model
    
    Lambda_pow = [ -10 ,-2,-1,0,(1:0.2:2), (2.1:0.1:2.5)]'; % column regularization constant; =0 means no regularization
    mu = 0.0;
    maxIterations = 1000; % maximum number of iterations
    log10tolerance = -6.0; % log base 10 of tolerance used to establish
    % convergence.
    numLambdaValues = length(Lambda_pow);
    stress_stack = [];
    models_stack = [];
    
    %% Main loop
    for m = 1 : numLambdaValues
        
        verbose = true; % print out progress information
        if ~verbose
            disp('Training... (please wait)')
        end
        if m==1
            lambda  = 0;
        else
            lambda = 10^ (Lambda_pow(m));
        end
        fprintf(1,'\n\n===> lambda=%f\n\n', lambda);
        
        % Run algorithm
        [StressValues, convergenceStatus, timeElapsedSeconds] = ...
            model.train(lambda, mu, maxIterations, ...
            log10tolerance, verbose);
        
        if verbose
            fprintf(1,'(in %.02f sec)\n',timeElapsedSeconds);
        end
        
        % Save the stress values
        U_valid = ones(Nv,Nv) - eye(Nv);
        Delta_valid = SIMDS.calcEuclideanDistanceMatrix(Xvalid);
        Delta_valid = Delta_valid + min(Delta_valid(Delta_valid>0))*0.001;
        Delta_valid = Delta_valid.*U_valid;
        stressOnValidation = model.stressOnValidation(model.C, model.Theta, Ktensor_valid, U_valid, Delta_valid);
        
        stress_stack = [stress_stack ; stressOnValidation];
        models_stack = [models_stack; model];
        %% save intermediate results
        lambda_str = num2str(lambda);
        save stackGrid_lambda stress_stack models_stack
        eval(['save model_lambda',lambda_str,' model']);
        
        %% Produce outputs (embedded training and testing samples)
        F_train = model.response(Ktensor);
        F_test = model.response(Ktensor_test);
        F_valid = model.response(Ktensor_valid);
        [colNorms, ~] = SIMDS.normsOfRowsAndColumns(model.C);
        eval(['save points_lambda',lambda_str,' F_train F_test F_valid colNorms']);
        %% Plot regularized cost (stress) versus iterations
        h = figure;
        set(h, 'Color', 'w')
        hold on
        if convergenceStatus == 1
            fmtString = 'iterations (converged in %.02f sec)';
        else
            fmtString = 'iterations (stopped after %.02f sec)';
        end
        xlabel(sprintf(fmtString, timeElapsedSeconds))
        ylabel('log_{10}(Reg. Stress)')
        plot((0:length(StressValues)-1), log10(StressValues))
        tempAdd=['tempadd',lambda_str,'.fig'];
        saveas(h,tempAdd)
        
        %  end
        %% plotbar chart the column norms vs. indices
        fh5 = figure;
        set(fh5, 'Color', 'w')
        axis([0 26 0 6])
        hold on
        bar(colNorms)
        tempAdd=['tempadd',lambda_str,'.fig'];
        saveas(fh5,tempAdd)
        %% Show plot of data at the end
        fh2 = figure;
        set(fh2, 'Color', 'w');
        hold on
        xlabel('f_1');
        ylabel('f_2');
        axis equal
        plot(F_test(:,1), F_test(:,2), 'b.')
        for i=1:N
            plot(F_train(i,1), F_train(i,2), 'o', 'MarkerSize', 7*1-(colNorms(i)/max(colNorms)), 'LineWidth', 2, 'MarkerFaceColor',1-(colNorms(i)/max(colNorms))*[1 1 1], 'MarkerEdgeColor', 'k');
            hold on
        end% [1 1 1] white, [0 0 0] black, larger ratio,larger colNorm, more important, blacker
        tempAdd=['tempadd',lambda_str,'.fig'];
        saveas(fh2,tempAdd)
        
        
        
        %% Print out remarks
        disp('Remember that Stress may have multiple local minima.')
        disp('So, each time you run it, you will probably get different results!')
        disp('Enjoy!')
        
    end % end m-loop
    % Plot the value of stress function of the data
    h1 = figure;
    set(h1, 'Color', 'w')
    hold on
    ylabel('log_{10}(Stress)')
    xlabel('\lambda')
    
    %%
    Lambda = [0; 10.^(Lambda_pow(2:end))];
    p = plot(Lambda, log10(stress_stack))%,'bo', 'MarkerFaceColor', 'b')
    p.Marker = '*';
    tempAdd=['tempadd.fig'];
    saveas(h1,tempAdd)
end % end i-loop