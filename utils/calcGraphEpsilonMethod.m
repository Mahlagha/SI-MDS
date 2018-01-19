function [GraphAdjacencyMatrix, isConnected] = calcGraphEpsilonMethod(X, epsilon)
%% calcGraphEpsilonMethod
%
% Computes a graph adjacency matrix given a set of samples. If the 
% Euclidean distance between a pair of samples is less than epsilon, the 
% pair is deemed connected.
%
% SYNTAX
%   [GraphAdjacencyMatrix, isConnected] = calcGraphEpsilonMethod(X, ...
%                                                              epsilon);
%
% INPUTS
%   X: NxD matrix containing N D-dimensional samples arranged in rows.
%   epsilon: positive scalar. Parameter of the epsilon-method for 
%   constructing a graph from a set of points.  
%
% OUTPUTS
%   GraphAdjacencyMatrix: NxN graph agency matrix produced via the 
%   epsilon-method. If the Euclidean distance between the m-th and n-th
%   samples are less or equal to epsilon, then GraphAdjacencyMatrix(m,n)=1;
%   otherwise =0. The matrix is symmetric.
%   isConnected: boolean scalar. If =1 (=0) the resulting graph is (not)
%   connected, i.e. it has more than one connected components.
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
    eval('help calcGraphEpsilonMethod')
    return
end

N = size(X,1);

%% Calculate matrix of pair-wise L_2 distances
D = quickL2DistanceMatrix(X);

%% Find pairs whose distance does not exceed epsilon
[idx1 idx2] = find(D <= epsilon);

P = length(idx1);

%% Populate adjacency matrix
GraphAdjacencyMatrix = zeros(N,N);
for p = 1 : P
    if idx1(p) ~= idx2(p)
        GraphAdjacencyMatrix(idx1(p), idx2(p)) = 1;
    end
end


%% Check if resulting graph is connected

nodeIDs = [];
for n = 1 : N-1
    for m = n+1 : N
        if GraphAdjacencyMatrix(m,n) == 1
            nodeIDs = [ nodeIDs; m; n];
        end
    end
end

isConnected = (length(unique(nodeIDs)) == N);


return % calcGraphEpsilonMethod()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [DM] = quickL2DistanceMatrix(X)
%% distanceMatrix
%
% Computes the pairwise L_2 (Euclidean) distance matrix between points of 
% X.
%
% SYNTAX
%   [DM] = quickL2DistanceMatrix(X);
%
%
% INPUTS
%   X: M-by-D matrix of M samples arranged in rows.
%
% OUTPUTS
%   DM: M-by-M matrix of pair-wise L_2 distances.
%
%
% NOTES
%   1) No validation of input arguments is performed.
%
% COPYRIGHT
%   (C) 2012 Georgios C. Anagnostopoulos
%   georgio@fit.edu
%

%% Initialization
M = size(X,1);

%% Compute the Grammian 
G = X * X';

%% Compute distance matrix
temp = diag(G) * ones(1,M);
DM = sqrt(temp + temp' - 2 * G);

return % distanceMatrix() 