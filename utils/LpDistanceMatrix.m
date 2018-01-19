function [DM] = LpDistanceMatrix(X1, X2, p)
%% LpDistanceMatrix
%
% Computes the pairwise L_p distance matrix between points of two sets.
%
% SYNTAX
%   [DM] = LpDistanceMatrix(X1, X2, p);
%
% INPUTS
%   X1:   M-by-D matrix of M samples arranged in rows.
%   X2:   N-by-D matrix of N samples arranged in rows.
%   p:    scalar >=1 used to compute the L_p norms. p=inf is allowed.
%
%
% OUTPUTS
%   DM:   M-by-N matrix of pair-wise L_p distances.
%
% NOTES
% 1) No validation of input arguments is performed.
% 2) Uses for-loops (not efficient).
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%

%% Initialization
M = size(X1,1);
N = size(X2,1);
DM = zeros(M,N);

%% Compute Lp pair-wise distances between sets X1 and X2
for m = 1 : M
    for n = 1 : N
        DM(m,n) = norm(X1(m,:)-X2(n,:), p);
    end
end

return % LpDistanceMatrix()