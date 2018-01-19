
function HD = hammingDistanceMatrix(X1,X2)
%% hammingDistanceMatrix
%
% Computes the matrix of Hamming distances between two sets of data.
%
% SYNTAX
%   [HD] = hammingDistanceMatrix(X1, X2);
%
% INPUTS
%   X1: M-by-D matrix of M samples arranged in rows.
%   X2: N-by-D matrix of N samples arranged in rows.
%
% OUTPUTS
%   HD: M-by-N matrix of pair-wise Hamming distances. HD(m,n) in {0,1,...,
%   D} provides the number of common feature values between the m-th sample
%   of X1 and the n-th sample of X2.
%
% NOTES
%   1. No input argument checking is being done!
%   2. Use of Hamming distances is meaningful, when each feature takes 
%   values from a small discrete set.  
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%

[N1, D] = size(X1);
N2 = size(X2, 1);

HD = zeros(N1, N2);
for n1 = 1 : N1
    for n2 = 1 : N2
        HD(n1,n2) = D - sum(X1(n1,:) == X2(n2,:));
    end
end

return % hammingDistanceMatrix()