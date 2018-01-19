function [KernelMatrix] = IMQKernelMatrix(X1, X2, spread, power)
%% IMQKernelMatrix
%
% Computes the inverse Multi-Quadric kernel matrix between two sets of data.
%
% SYNTAX
%   [K] = IMQKernelMatrix(X1,X2,spread,power);
%
% INPUTS
%   X1: M-by-D matrix of M samples arranged in rows.
%   X2: N-by-D matrix of N samples arranged in rows.
%   spread: (positive) scalar; parameter of the kernel.
%   power: positive scalar; parameter of the kernel.
%
% OUTPUTS
%   K: M-by-N kernel matrix.
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
    eval('help IMQKernelMatrix')
    return
end

%% Initializations
[M,D] = size(X1);
[N,D] = size(X2);

%% Compute squared Euclidean distances between sets X1 and X2
MatrixOfSquaredDistances = ...
    sum(X1 .* X1, 2) * ones(1,N) - 2.0 * X1 * X2' + ...
    ones(M,1) * (sum(X2 .* X2, 2))';

%% Compute kernel matrix
K = (1.0 + MatrixOfSquaredDistances / spread^2) .^ (-power);


return % IMQKernelMatrix()