
function HD = hammingDistanceMatrixNonZero(X1,X2,self)
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
%   self: a boolen variable to indicate if the distance is calculated bw an
%   object and itslef or not.
% OUTPUTS
%   HD: M-by-N matrix of pair-wise Hamming distances. HD(m,n) in {0,1,...,
%   D} provides the number of common feature values between the m-th sample
%   of X1 and the n-th sample of X2. But it ignores the zeros.
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
        HD(n1,n2) = D - sum(X1(n1,:) == X2(n2,:) & X1(n1,:)>0);% ignore zero elements
    end
end
% if the distance is between an object and itself, Zeros in diagonal should not be ignored, the diagonals should be zero
if(self)
 U = ones(N1,N1) - eye(N1);
 HD = HD.*U;
end


return % hammingDistanceMatrix()