function K = hammingKernelMatrix(X1, X2, ksi)
%% hammingKernelMatrix
%
% Computes the Hamming kernel matrix between two sets of data.
%
% SYNTAX
%   [K] = hammingKernelMatrix(X1, X2, ksi);
%
% INPUTS
%   X1: M-by-D matrix of M samples arranged in rows.
%   X2: N-by-D matrix of N samples arranged in rows.
%   ksi: scalar in (0,1); parameter of the kernel.
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
    eval('help hammingKernelMatrix')
    return
end

%% Compute Hamming distances
HD = hammingDistanceMatrix(X1, X2);

%% Compute kernel matrix
K = (ksi * ones(size(HD))) .^ (HD);

return % hammingKernelMatrix()
