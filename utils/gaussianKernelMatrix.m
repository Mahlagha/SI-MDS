function [K] = gaussianKernelMatrix(X1,X2,sigmaSquared)
%% gaussianKernelMatrix
%
% Computes the Gaussian kernel matrix between two sets of data.
%
% SYNTAX
%   [K] = gaussianKernelMatrix(X1,X2,sigmaSquared);
%
% INPUTS
%   X1: M-by-D matrix of M samples arranged in rows.
%   X2: N-by-D matrix of N samples arranged in rows.
%   sigmaSquared: positive scalar. Scaling parameter for the Gaussian
%   kernel.
%
% OUTPUTS
%   K: M-by-N kernel matrix, where K(m,n) = k(x1_m, x2_n; sigmaSquared).
%
% NOTES
%   1. No input argument checking is being done!
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%

%% Display help text, if no arguments are provided
if nargin == 0
    eval('help gaussianKernelMatrix')
    return
end

%% Compute pair-wise Euclidean distances between points of X1 and X2. 
L2DistancesSquared = LpDistanceMatrix(X1, X2, 2) .^ 2;

%% Compute Gaussian kernel matrix of X1 vs. X2
K = exp(- L2DistancesSquared / sigmaSquared);

return % gaussianKernelMatrix()
