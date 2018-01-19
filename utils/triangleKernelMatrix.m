function K = triangleKernelMatrix(X1, X2, spread, power)
%% triangleKernelMatrix
%
% Computes the triangular kernel matrix between two sets of data.
%
% SYNTAX
%   [K] = polyKernelMatrix(X1,X2,spread,power);
%
% INPUTS
%   X1: M-by-D matrix of M samples arranged in rows.
%   X2: N-by-D matrix of N samples arranged in rows.
%   spread: non-negative scalar spread; parameter of the kernel.
%   power: positive integer; parameter of the kernel.
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
    eval('help triangleKernelMatrix')
    return
end
 
%% Compute pair-wise Euclidean distances between sets X1 and X2. 
D = sqrt(sum(X1 .* X1, 2) * ones(1,N) - 2.0 * X1 * X2' + ...
    ones(M,1) * (sum(X2 .* X2, 2))');

%% Compute kernel matrix
K =  (1.0 - D / spread) .^ power;
idx = find(K < 0);
K(idx) = 0.0;
 
return

