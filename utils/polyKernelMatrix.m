function K = polyKernelMatrix(X1, X2, degree, useOffset)
%% polyKernelMatrix
%
% Computes the polynomial kernel matrix between two sets of data.
%
% SYNTAX
%   [K] = polyKernelMatrix(X1,X2,degree,useOffset);
%
% INPUTS
%   X1: M-by-D matrix of M samples arranged in rows.
%   X2: N-by-D matrix of N samples arranged in rows.
%   degree: non-negative scalar. degree=1.0 and useOffset=false yields the 
%   ordinary Gram matrix between X1 and X2.
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
    eval('help polyKernelMatrix')
    return
end

%% Compute Gram matrix
G = X1 * X2';

%% Compute kernel matrix
if useOffset == 0
    K = G .^ degree;
else
    K = (G + 1.0) .^ degree;
end

return % polyKernelMatrix()
