function [StressValues, convergenceStatus, timeElapsedSeconds] = ...
    train(obj, lambda_, mu_, maxIter, log10tol, verbose)
%% train
%
% Public method of SIMDS class that implements the Iterative Majorization
% (IM) algorithm for training a SIMDS model.
%
% SYNTAX
%   [StressValues, convergenceStatus, timeElapsedSeconds] = ...
%                    train(obj, lambda_, mu_, maxIter, log10tol, verbose);
%
% INPUTS
%   lambda_: non-negative scalar used for the penalization of column-norms
%   of the weight matrix C.
%   mu_: non-negative scalar used for the penalization of row-norms
%   of the weight matrix C.
%   maxIter: positive integer for the maximum number of iterations to be
%   performed by the algorithm.
%   log10tol: real scalar use to establish convergence of the algorithm.
%   Convergence is established, when the changes in C and Theta become too
%   small.
%   verbose: boolean. If true, the algorithm produces screen output at
%   every iteration about its progress.
%
% OUTPUTS
%   StressValues: (T+1)-dimensional vector of (regularized) stress values
%   for each iteration of the algorithm, assuming that training concluded
%   after T iterations. StressValues(1) contains the initial (regularized)
%   stress value
%   convergenceStatus: boolean; if true, the algorithm managed to converge.
%   timeElapsedSeconds: non-negative scalar containing the training's
%   time duration in seconds.
%
% NOTES
%   1. No input argument checking is being done!
%   2. Before exiting, the method updates the C and Theta public properties
%   of the object. If the algorithm has converged, these values are the
%   optimal parameters achieved by the training.
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start timer
startTime = cputime;

% disable cvx output
cvx_quiet(true)

% setup private properties (regularization  and proximal constants)
obj.lambda = lambda_;
obj.mu = mu_;
%obj.proxLambda = proxLambda_;

% Setup initial parameter values obtained when the model was initialized
C_init = obj.C;
Theta = obj.Theta;


% Other setups
J = obj.J;

% Calculate & print out initial regularized cost
RegularizedCost_init = obj.regularizedStress(C_init, Theta);
if verbose
    fprintf(1,'#%04d\tlog(RC)=%f\n', 0, log10(RegularizedCost_init));
end

C_prime = C_init;
RegularizedCost_prime = RegularizedCost_init;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Main Loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Variable initialization
iter = 0;
notConverged = true;
convergenceStatus = false;
StressValues = [ RegularizedCost_init ];

while notConverged
    
    iter = iter + 1;
    
    %% Update C %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%
    if  isnan(Theta)
        fprintf(1, 'breakpoint...\n\n');
    end
    [vecC, grad] = obj.vecIMupdateCProx_accelerated(Theta, C_prime, Theta, 3000, log10tol, true );
    %%%
    
    % Unvectorize vecC
    C = SIMDS.ivec(vecC, obj.P);
    
    % Setup for backtracking
    stepLength = 1.0;
    C_trial = C;
    trialsC = 1;
    maxTrialsC = 300;
    regularizedStress_prime = obj.regularizedStress(C_prime, Theta);
    
    
    %%%%%%TO DO: if backtracking is needed or not?%%%%%%%%%%
    % Backtracking loop for robustness
    while obj.regularizedStress(C_trial, Theta) > ...
            regularizedStress_prime
        
        
        % reduce step length
        stepLength = 0.9 *  stepLength;
        
        % new trial C value
        C_trial = (1.0 - stepLength) * C_prime + stepLength * C;
        
        if trialsC > maxTrialsC
            
            % Unable to find a better value for C.
            % In this case, C will remain unchanged.
            break
        end
        
        trialsC = trialsC + 1;
        
    end % C backtracking while-loop
    
    % Finalize C
    C = C_trial;
    %we have to change this psi vector since we've removed CVX and theta is fixed now
    
    %% Print progress information %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    PsiVector_prime = SIMDS.vec(C_prime);
    PsiVector =  SIMDS.vec(C);
    
    log10maxAbsDiff = log10(norm(PsiVector - PsiVector_prime, inf));
    
    % calculate newly attained regularized stress
    RegularizedCost = regularizedStress(obj, C, Theta);
    
    if verbose
        fprintf(1,'#%04d\tlog(RC)=%f\tlog(Dpsi)=%f\t(trials: %d, %d)\tlog(grad)=%f\n',...
            iter, log10(RegularizedCost), log10maxAbsDiff, trialsC, ...
            1, log10(norm(grad)));
    end
    
    
    
    %% Check for convergence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Check if the algorithm has converged
    if log10maxAbsDiff <= log10tol
        notConverged = false;
        
        if verbose
            fprintf(1, 'CONVERGED!\n\n');
        end
        
        StressValues = [StressValues; RegularizedCost];
        break
    end
    
    % Check if the algorithm has exhausted the maximum number of iterations
    if iter >= maxIter & notConverged
        
        if verbose
            fprintf(1, 'STOPPED (exceeded maximum itearions)...\n\n');
        end
        
        StressValues = [StressValues; RegularizedCost];
        break
    end
    
    % Check if the algorithm must be halted due to numerical errors
    % which cause oscillations
    if RegularizedCost_prime < RegularizedCost
        
        % Algorithm needs to be halted, as the regularized stress
        % increased in the previous iteration.
        
        % Use previous variable values
        C = C_prime;
        
        
        if verbose
            fprintf(1, 'HALTED!\n\n');
        end
        
        break
    else
        
        % if we did not need to halt the algorithm, update variables
        C_prime = C;
        if  isnan(Theta)
            fprintf(1, 'breakpoint...\n\n');
        end
        RegularizedCost_prime = RegularizedCost;
        StressValues = [StressValues; RegularizedCost];
        
        %         %%%
        %         y_prime = y;
        %         eta_prime = eta;
        %         %%%
        
    end
    
    % we haven't changed theta, just to double check
    fprintf(1,'Theta=%f\n', obj.Theta);
end % main while-loop



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Epilogue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Update properties
obj.C = C;
%obj.Theta = Theta;
% Populate return arguments
convergenceStatus = ~notConverged;
timeElapsedSeconds = cputime - startTime;

% re-enable CVX outputs
cvx_quiet(false)


return % train()