classdef SIMDS < handle
% SIMDS class    
%
% Class that implements the training and performance phases of the
% Regularized Multi-Kernel Learning Least-Squares MDS (SIMDS) model.
%
% PUBLIC PROPERTIES
%
%   Ktensor
%    NxNxJ kernel tensor consisting of J NxN kernel matrices to be used for
%    Multi-Kernel Learning (MKL). N is the number of training samples.
%   C
%    PxN weight matrix used for the resulting embedding. P is the embedding
%    dimension.
%   Theta
%    J-dimensional vector to hold the MKL coefficients.
%   Delta
%    NxN matrix of pair-wise dissimilarities between training data samples.
%   U
%    NxN matrix weighing the importance of discrepancies between
%    dissimilarities and the corresponding Euclidean distances in the
%    embedding space.
%   
%             
% PUBLIC METHODS
%
%   SIMDS()
%    Constructs an object of the SIMDS class (an SIMDS model) and populates
%    public and private properties of the object.
%   initialize()
%    Resets/recalculates public and private properties of the object.
%   response()
%    Computes the image of input data in the embedding space.
%   responseCentered()
%    Computes the mean-centered image of input data in the embedding space.
%   train()
%    Uses the standard Iterative Majorization (IM) algorithm to compute the
%    optimal embedding.

%   
% NOTES
%   1. No input argument checking is being done in any method!
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%
    
    %% Public Properties
    properties
        Ktensor = []; % Kernel tensor evaluated on input data
        C = []; % Weight matrix
        Theta = []; % MKL coefficients
        Delta = []; % Delta matrix containing the pair-wise dissimilarities
        U = []; % U matrix
    end % properties
    
    %% Public Methods
    methods
        
        % Constructor 
        function obj = SIMDS(Ktensor_, C_, Theta_, Delta_, U_)
            if  nargin > 0
                obj = initialize(obj, Ktensor_, C_, Theta_, Delta_, U_);
            else
                 eval('help SIMDS')
            end
        end % SIMDS()
        
        % Object (re)initialization
        obj = initialize(obj, Ktensor_, C_, Theta_, Delta_, U_)
        
        % Compute model's output
        function F = response(obj, Ktensor_)
           F = SIMDS.responseCTheta(Ktensor_, obj.C, obj.Theta); 
        end
        
        % Compute model's output that is mean-centered 
        function F = responseCentered(obj, Ktensor_)
            F = SIMDS.responseCTheta(Ktensor_, obj.C, obj.Theta);
            N_ = size(F,1);
            F = (eye(N_) - ones(N_,N_) / double(N_)) * F;
        end
       
        % Train model with standard IM algorithm
        [StressValues, convergenceStatus, timeElapsedSeconds] = ...
    train(obj, lambda_, mu_, maxIter, log10tol, verbose)

         %%%
        % Helper function stressOnValidation()
        function value = stressOnValidation(obj, C_, Theta_, Ktensor_, U, Delta)
            [F_valid , SB_valid] = obj.calcFandSBGeneral(C_, Theta_, Ktensor_, U, Delta);
            SA_valid = SIMDS.laplacian(U);
            constant_valid = 0.5 * sum(sum( U .* Delta .* Delta ));
            value = trace( F_valid'*(SA_valid-2*SB_valid)*F_valid) + constant_valid;
            
            % Guard against numerical errors, when stress is supposed
            % to be equal to zero, as in the case of perfect embedding.
            value = max(value, 0.0);
        end % end stressOnValidation
    
        %%%
                                        
    end % methods(Access = public)

    
    

    
    %% Private Properties
    properties(GetAccess = private, SetAccess = private)
        
        N = 0; % number of training samples
        J = 0; % number of kernels partaking in MKL setup
        P = 0; % dimension of the embedding space
        
        constant = []; % constant term of cost (stress) function
        SA = []; % the $\mathbf{S}_A$ matrix
        UDelta = []; % Hadamard product of U with Delta
        Ip = []; % PxP identity matrix
        In = []; % NxN identity matrix
        
        lambda = 0.0; % regularization constant for columns of C
        mu = 0.0; % regularization constant for rows of C
        
        %%%
        proxLambda= 10^(-6);%the proximal operator constant as well as the...
                        % step length of the gradient method.
       
        %%%
       
    end % properties(GetAccess = private, SetAccess = private)
    
    
    
    %% Public Static Methods
    methods(Static, Access = public)
        
        % Helper function calcEuclideanDistanceMatrix()
        function D = calcEuclideanDistanceMatrix(X)
            M = size(X,1);
            G = X * X';
            temp = diag(G) * ones(1,M);
            D = real(sqrt(temp + temp' - 2 * G));
        end % calcEuclideanDistanceMatrix()
        
        % Helper function normalize()
        function N = normalize(X)
            N = (X-repmat(mean(X),size(X,1),1))./repmat(std(X),size(X,1),1);
        end% normalize()
        
        % Helper function normsOfRowsAndColumns()
        function [vColumnNorms, vRowNorms] = normsOfRowsAndColumns(X)
            temp = X .* X;
            vColumnNorms = sqrt(sum(temp, 1))';
            vRowNorms = sqrt(sum(temp, 2));
        end % normsOfRowsAndColumns()
        
        % Helper function ivec()
        function Matrix = ivec(vectorizedMatrix, rows)
            Matrix = reshape(vectorizedMatrix,rows, ...
                length(vectorizedMatrix)/rows);
        end% ivec
        
              %%%
        % Helper function calcFandSBGeneral()
        function [F_, SB_] = calcFandSBGeneral(C_, Theta_, Ktensor, U, Delta)
            if  isnan(Theta_)
                fprintf(1, 'breakpoint...\n\n');
            end
            
            % Compute model response
            F_ = SIMDS.responseCTheta(Ktensor, C_, Theta_);
            if  isnan(F_)
                fprintf(1, 'breakpoint...\n\n');
            end
            
            % Compute intermediate quantities
            D = SIMDS.calcEuclideanDistanceMatrix(F_);
            if  isnan(D)
                fprintf(1, 'breakpoint...\n\n');
            end
            Dcross = SIMDS.crossInversion(D);
            if  isnan(Dcross)
                fprintf(1, 'breakpoint...\n\n');
            end
            UDelta = U .* Delta;
            Z = real(UDelta .* Dcross); % added real() for debugging purposes
            
            % Compute SB
            SB_ = SIMDS.laplacian(Z);
        end
        
        %%% end calcFandSBGeneral
        
    end%%end Public Static helpers
    %% Private Static Helper Methods
    methods(Static, Access = private)
 
        % Helper function vec()
        function vectorizedMatrix = vec(Matrix)
            vectorizedMatrix = Matrix(:);
        end
        
        % Helper function laplacian()
        function LaplacianMatrix = laplacian(SymmetricMatrix)
            LaplacianMatrix = diag(sum(SymmetricMatrix)) - SymmetricMatrix;
        end
        
        % Helper function calcK()
        function K = calcK(Ktensor_, Theta_)
            [M,N,J] =size(Ktensor_);
            K = zeros(M,N);
            for j = 1 : J
                K = K + Theta_(j) * Ktensor_(:,:,j);
            end % j-loop
        end
        
        % Helper function responseCTheta()
        function F = responseCTheta(Ktensor_, C_, Theta_)
            K = SIMDS.calcK(Ktensor_, Theta_);
            F = K * C_';
        end
        
        % Helper function crossInversion()
        function Dsci = crossInversion(D)
            D(abs(D) < 1e-08) = 0.0;
            Dsci = D .^ (-1);
            Dsci(Dsci == inf) = 0.0;
        end % crossInversion()
 
    end % methods(Static, Access = private)
        
    
    %% Private Methods
    methods(Access = private)
        
        % Helper function 
        function [Lc, Lr] = calcLcLr(obj, C_)
            [vColumnNorms, vRowNorms] = ...
                SIMDS.normsOfRowsAndColumns(C_);
            Lc =  diag(vColumnNorms .^ (-1));
            Lr =  diag(vRowNorms .^ (-1));            
        end
        
        % Helper function calcL()
        function L_prime = calcL(obj, C_prime_)
            [Lc_prime, Lr_prime] = obj.calcLcLr(C_prime_);

            L_prime = obj.lambda * kron(Lc_prime, obj.Ip) + ...
                obj.mu * kron(obj.In, Lr_prime);
        end
        
        % Helper function calcFandSB()
        function [F_, SB_] = calcFandSB(obj, C_, Theta_)
            if  isnan(Theta_)
                fprintf(1, 'breakpoint...\n\n');
            end
            
            % Compute model response
            F_ = SIMDS.responseCTheta(obj.Ktensor, C_, Theta_);
            if  isnan(F_)
                fprintf(1, 'breakpoint...\n\n');
            end
            
            % Compute intermediate quantities
            D = SIMDS.calcEuclideanDistanceMatrix(F_);
            if  isnan(D)
                fprintf(1, 'breakpoint...\n\n');
            end
            Dcross = SIMDS.crossInversion(D);
            if  isnan(Dcross)
                fprintf(1, 'breakpoint...\n\n');
            end
            Z = real(obj.UDelta .* Dcross); % added real() for debugging purposes
            
            % Compute SB
            SB_ = SIMDS.laplacian(Z);
        end
        
  
        
        % Helper function calcGandh()
        function [G, h] = calcGandh(obj, C_, Theta_prime_)
            
            % Setup
            J = obj.J;
            G = zeros(J,J);
            h = zeros(J,1);
            K = SIMDS.calcK(obj.Ktensor, Theta_prime_);
            [notUsed, SB] = obj.calcFandSB(C_, Theta_prime_);
            
            % Populate G and h
            for i = 1 : J
                tmp = C_ * obj.Ktensor(:,:,i);
                h(i,1) = trace(tmp * SB * K * C_');
                for j = 1 : i
                    G(i,j) =trace(tmp * obj.SA * obj.Ktensor(:,:,j) * C_');
                    G(j,i) = G(i,j);
                end % j-loop
            end % i-loop
        end
        
        % Helper function vecIMupdateC()
        function vecC = vecIMupdateC(obj, C_prime, Theta_prime)
            % Computes the IM update for C and returns it as a vector
            
            % Vectorize C_prime
            vecC_prime = SIMDS.vec(C_prime);
            
            % Setup matrices
            K_prime = SIMDS.calcK(obj.Ktensor, Theta_prime);
            KSAK_prime = K_prime * obj.SA * K_prime;
            [notUsed, SB_prime] = obj.calcFandSB(C_prime, Theta_prime);
            KSBK_prime = K_prime * SB_prime * K_prime;
            L_prime = obj.calcL(C_prime);
            
            % Compute candidate IM update for C
            vecC = pinv( kron(KSAK_prime, obj.Ip) + 0.5 * L_prime) * ...
                ( kron(KSBK_prime, obj.Ip) * vecC_prime);
        end
        
        
        %%%
         % Helper function regRow()
        function value = regRow(obj, C_)
            [notUsed, vRowNorms] = SIMDS.normsOfRowsAndColumns(C_);
            value = obj.mu * sum(vRowNorms);
        end
        
         % Helper function regCol()
        function value = regCol(obj, C_)
            [vColumnNorms, notUsed] = SIMDS.normsOfRowsAndColumns(C_);
            value = obj.lambda * sum(vColumnNorms) ;
        end
         % Helper function proxRegRow()
        function value = proxRegRow(obj, V_)
            [notUsed, vRowNorms] = SIMDS.normsOfRowsAndColumns(V_);
            value = (max((1-((obj.proxLambda*obj.mu* ones(size(V_,1),size(V_,2)))./ ...
                repmat(vRowNorms,1,size(V_,2)))),0)) .*V_;
        end
         
          % Helper function proxRegCol()
        function value = proxRegCol(obj, V_)
            [vColumnNorms, notUsed] = SIMDS.normsOfRowsAndColumns(V_);
            value = (max((1-((obj.proxLambda*obj.lambda* ones(size(V_,1),size(V_,2)))./ ...
                (repmat(vColumnNorms,1,size(V_,1)))')),0)) .*V_;
        end
        
        
          % Helper function proxAvg()
        function value = proxAvg(obj, V_)
            temp1 = obj.proxRegRow(V_);
            temp2 = obj.proxRegCol(V_);
            value = temp1+temp2;
        end
        
        
         % Helper function gradMajorizedStressC()
        function vecGrad = gradMajorizedStressC(obj, C_, Theta_,C_prime, Theta_prime)
            % Computes the gradient of the stress
            % with respect to vec(C)
            if  isnan(Theta_prime)
                fprintf(1, 'breakpoint...\n\n');
            end
            if  isnan(Theta_)
                fprintf(1, 'breakpoint...\n\n');
            end
            
            % Vectorize C_
            vecC_ = SIMDS.vec(C_);
            % Vectorize C_prime
            vecC_prime = SIMDS.vec(C_prime);
            
            % Setup matrices
            K = SIMDS.calcK(obj.Ktensor, Theta_);
           % [notUsed, SB] = obj.calcFandSB(C_, Theta_);
            K_prime = SIMDS.calcK(obj.Ktensor, Theta_prime);
            [notUsed, SB_prime] = obj.calcFandSB(C_prime, Theta_prime);
            A = kron((K*obj.SA*K), obj.Ip);
            B = kron((K*SB_prime*K_prime), obj.Ip)*vecC_prime;
            % Return gradient vector 
            vecGrad = 2*A*vecC_ - 2*B;
        
        end
                
         % Helper function vecIMupdateCProx_accelerated() Accelerated version
         % This grad output is added for test purposes, you can remove it
        function [vecC, grad] = vecIMupdateCProx_accelerated(obj, Theta_, C_prime, Theta_prime, maxIter, log10tol, verbose)
            % Computes the IM update for C and returns it as a vector with
            % Accelerated proximal average method
            
            C = zeros(obj.P,obj.N, maxIter);
            Z = zeros(obj.P,obj.N, maxIter);
            Y = zeros(obj.P,obj.N, maxIter);
            eta = zeros(maxIter);
            
            C(:,:,1) = C_prime;
            Y(:,:,2) = C(:,:,1);
            eta(2) = 1;
            notConverged = true;
            
            for t = 2:maxIter
                %% Update Z
                if  isnan(Theta_prime)
                    fprintf(1, 'breakpoint...\n\n');
                end
                if  isnan(Theta_)
                    fprintf(1, 'breakpoint...\n\n');
                end
                grad = obj.gradMajorizedStressC(Y(:,:,t), Theta_,C_prime, Theta_prime);
                if  isnan(grad)
                    fprintf(1, 'breakpoint...\n\n');
                end
                vecYTemp = SIMDS.vec(Y(:,:,t));
                vecZTemp = vecYTemp-obj.proxLambda*grad;
                Z(:,:,t) = SIMDS.ivec(vecZTemp,obj.P);
                if  isnan(Z)
                    fprintf(1, 'breakpoint...\n\n');
                end
                
                %% Compute matrix C
                C(:,:,t) =0.5* obj.proxAvg(Z(:,:,t));
                if  isnan(C)
                    fprintf(1, 'breakpoint...\n\n');
                end
                
                %% Compute eta
                eta(t+1) = (1+sqrt(1+(4*eta(t)^2)))/2;
                if  isnan(eta)
                    fprintf(1, 'breakpoint...\n\n');
                end
                
                %% Compute matrix Y
                vecCTemp_new = SIMDS.vec(C(:,:,t));
                vecCTemp_old = SIMDS.vec(C(:,:,t-1));
                vecYTemp = vecCTemp_new + ((eta(t)-1)/eta(t+1))*(vecCTemp_new - vecCTemp_old);
                Y(:,:,t+1) = SIMDS.ivec(vecYTemp,obj.P);
                
                %% check for convergence
                log10maxAbsDiff = log10(norm(C(:,:,t) - C(:,:,t-1), inf));
                %%%%%DEBUGING
                if t ==2
                    fprintf(1, 'log10normC = %f\tlog10normDiff = %f\n', log10(norm(C(:,:,t))), log10maxAbsDiff);
                end
                
                if log10maxAbsDiff <= log10tol
                    notConverged = false;
                    if verbose
                        fprintf(1, 'Proximal Gradient CONVERGED!\n\n');
                    end
                   % vecC = SIMDS.vec(C(:,:,t));
                    break
                end
            end
            if verbose && notConverged
                fprintf(1, 'Proximal Gradient STOPPED (exceeded maximum itearions)...\n\n');
            end
            vecC = SIMDS.vec(C(:,:,t));
            
             
        end % end vecIMupdateCProx_accelerated()
        
         % Helper function vecIMupdateCProx()

        function [vecC, grad] = vecIMupdateCProx(obj, Theta_, C_prime, Theta_prime, maxIter, log10tol, verbose)
            % Computes the IM update for C and returns it as a vector with
            %proximal average method
            
            C = zeros(obj.P,obj.N, maxIter);
            Z = zeros(obj.P,obj.N, maxIter);
            C(:,:,1) = C_prime;
            notConverged = true;
            
            for t = 2:maxIter
                %% Update Z
                if  isnan(Theta_prime)
                    fprintf(1, 'breakpoint...\n\n');
                end
                if  isnan(Theta_)
                    fprintf(1, 'breakpoint...\n\n');
                end
                grad = obj.gradMajorizedStressC(C(:,:,t-1), Theta_,C_prime, Theta_prime);
                vecCTemp = SIMDS.vec(C(:,:,t-1));
                vecZTemp = vecCTemp-obj.proxLambda*grad;
                Z(:,:,t) = SIMDS.ivec(vecZTemp,obj.P);
                
                %% Compute matrix C
                C(:,:,t) =0.5* obj.proxAvg(Z(:,:,t));
                
                %% check for convergence
                log10maxAbsDiff = log10(norm(C(:,:,t) - C(:,:,t-1), inf));
                %%%%%DEBUGING
                if t ==2
                    fprintf(1, 'log10normC = %f\tlog10normDiff = %f\n', log10(norm(C(:,:,t))), log10maxAbsDiff);
                end
                
                if log10maxAbsDiff <= log10tol
                    notConverged = false;
                    if verbose
                        fprintf(1, 'Proximal Gradient CONVERGED!\n\n');
                    end
                   % vecC = SIMDS.vec(C(:,:,t));
                    break
                end
            end
            if verbose && notConverged
                fprintf(1, 'Proximal Gradient STOPPED (exceeded maximum itearions)...\n\n');
            end
            vecC = SIMDS.vec(C(:,:,t));




        end % end vecIMupdateCProx()
                
        
        
        %%%
        
        % Helper function gradRegStressC()
        function vecGrad = gradRegStressC(obj, C_, Theta_)
            % Computes the gradient of the regularized stress
            % with respect to vec(C)
            
            % Setup matrices
            K = SIMDS.calcK(obj.Ktensor, Theta_);
            [notUsed, SB] = obj.calcFandSB(C_, Theta_);
            [Lc, Lr] = obj.calcLcLr(C_);
            
            temp1 = C_ * (K * (obj.SA - SB) * K + obj.lambda * Lc);
            temp2 = obj.mu * C_' * Lr;
            
            % Return gradient vector
            vecGrad = SIMDS.vec(temp1) + SIMDS.vec(temp2);
        end
        
        % Helper function gradRegStressCapprox()
        function vecGrad = gradRegStressC2(obj, C_, Theta_)
            % Computes the approx. gradient of the regularized stress
            % with respect to vec(C). Primarily for debugging purposes.
            
            ER = obj.regularizedStress(C_, Theta_);
            Grad = zeros(size(C_));
            epsilon = 10e-04;
            
            for p = 1 : obj.P
                for n = 1 : obj.N
                    C_pertrubed = C_;
                    C_pertrubed(p,n) = C_pertrubed(p,n) + epsilon;
                    ER_pertrubed = obj.regularizedStress(C_pertrubed, ...
                        Theta_);
                    Grad(p,n) = (ER_pertrubed - ER) / epsilon;
                end % n-loop
            end % p-loop
                      
            % Return gradient vector
            vecGrad = SIMDS.vec(Grad);
        end
        
        % Helper function stress()
        function value = stress(obj, C_, Theta_)
            [F, SB] = obj.calcFandSB(C_, Theta_);
            value = trace(F' * (obj.SA - 2.0 * SB) * F) + obj.constant; 
            
            % Guard against numerical errors, when stress is supposed
            % to be equal to zero, as in the case of perfect embedding.
            value = max(value, 0.0);
        end
       
               
        % Helper function penalty()
        function value = penalty(obj, C_)
            [vColumnNorms, vRowNorms] = SIMDS.normsOfRowsAndColumns(C_);
            value = obj.lambda * sum(vColumnNorms) + obj.mu * ...
                sum(vRowNorms);
        end
        
        % Helper function regularizedStress()
        function value = regularizedStress(obj, C_, Theta_)
            value = obj.stress(C_, Theta_) + obj.penalty(C_);
        end % regularizedStress()
        
        % Helper function stressMajorizer()
        function value = stressMajorizer(obj, C_, Theta_, C_prime_, ...
                Theta_prime_)
            F = SIMDS.responseCTheta(obj.Ktensor, C_, Theta_);
            [F_prime, SB_prime] = obj.calcFandSB(C_prime_, Theta_prime_);
            
            value = trace(F' * obj.SA * F) - 2.0 * ...
                trace(F_prime' * SB_prime * F) + obj.constant;
            
            % Guard against numerical errors, when stress is supposed
            % to be equal to zero, as in the case of perfect embedding.
            value = max(value, 0.0);            
        end % stressMajorizer()
        
        %%%
        % Helper function stressMajorizerReg()
        % It's the stress majorizer + regulizers themselves
        function value = stressMajorizerReg(obj, C_, Theta_, C_prime_, ...
                Theta_prime_)
            value = obj.stressMajorizer(C_, Theta_, ...
                C_prime_, Theta_prime_) + obj.penalty(C_);
        end % regularizedStress()
        
        
        %%%
        
        % Helper function penaltyMajorizer()
        function value = penaltyMajorizer(obj, C_, C_prime_)
            [vColumnNorms, vRowNorms] = ...
                SIMDS.normsOfRowsAndColumns(C_prime_);
            Lc_prime =  diag(vColumnNorms .^ (-1));
            Lr_prime =  diag(vRowNorms .^ (-1));
            value = 0.5 * ( obj.lambda * trace(C_ * Lc_prime * C_') + ...
                obj.mu * trace(C_' * Lr_prime * C_) + ...
                obj.penalty(C_prime_) );
        end % penaltyMajorizer()
        
        % Helper function regularizedStressMajorizer()
        function value = regularizedStressMajorizer(obj, C_, Theta_, ...
                C_prime_, Theta_prime_)
            value = obj.stressMajorizer(C_, Theta_, ...
                C_prime_, Theta_prime_) + ...
                obj.penaltyMajorizer(C_, C_prime_);
        end
                
    end % methods(Access = private)
        
        
        
end % classdef SIMDS
    
    
    

%% NOTES
% The constructor method must be defined within the classdef block and,
% therefore, cannot be in a separate file. 
%
% Set and get property access methods must be defined within the classdef
% block and, therefore, cannot be in separate files.
%
% Nonstatic methods must include an explicit object variable in the
% function definition. 
% 
% You need to inherit from the "handle" class, so that changes to the state
% of the object are saved.
