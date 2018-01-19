function [Delta] = calcGraphDistance(DistanceMatrix, GraphAdjacencyMatrix)
%% calcGraphDistance
%
% Based on a (square) matrix of pair-wise distances between points and a
% (square) graph adjacency matrix between pairs of the same points, the 
% function outputs the matrix of shortest pair-wise graph distances between 
% points based on the element-wise (Hadamard) product of the two 
% aforementioned matrices.
%
% SYNTAX
%   [Delta] = calcGraphDistance(DistanceMatrix, GraphAdjacencyMatrix);
%
% INPUTS
%   DistanceMatrix: NxN matrix containing pair-wise distances between N
%   samples.
%   GraphAdjacencyMatrix: NxN graph adjacency matrix for the aformentioned
%   set of N points, which serve as edges of a graph. If
%   GraphAdjacencyMatrix(m,n)=1 or 0, there is (not) an edge between the
%   m-th and n-th sample.
%
% OUTPUTS
%   Delta: NxN graph shorthest distance matrix. It contains the shorthest 
%   graph distances between samples as computed by Dijkstra's algorithm
%   the data in DistanceMatrix .* GraphAdjacencyMatrix. Delta will be
%   symmetric.
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
    eval('help calcGraphDistance')
    return
end

%% Initialization
N = size(DistanceMatrix, 1);

%% Construct GraphDistanceMatrix
GraphDistanceMatrix = DistanceMatrix .* GraphAdjacencyMatrix;

% Indicate no edge by replacing 0 with inf
idx = find(GraphDistanceMatrix == 0.0);
GraphDistanceMatrix(idx) = inf;

% Zero out diagonal elements
i = (1 : N); 
idx = (N+1)*(i-1) + 1;
GraphDistanceMatrix(idx) = 0.0;


%% Find all pair-wise graph distances
Delta = [];

for n = 1 : N
    sourceNode = n;
    distanceVector = dijkstra(GraphDistanceMatrix, sourceNode);
    Delta = [ Delta distanceVector ];
end



end % calcGraphDistance()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function distanceVector = dijkstra(GraphDistanceMatrix, sourceNode)
%
% distanceVector = dijkstra(GraphDistanceMatrix, sourceNode)
% 

N = size(GraphDistanceMatrix, 1);


%% Initializations
distanceVector = inf * ones(N,1);
%previousNodeID = -1 * ones(N, 1);
distanceVector(sourceNode) = 0.0;
Q = (1:N);

%% Main loop
while ~isempty(Q)
    
    lenQ = length(Q);
    
    u = -1;
    min_dist = inf;
    for q = 1 : lenQ
       if  distanceVector(Q(q)) < min_dist
           min_dist = distanceVector(Q(q));
           u = Q(q);
       end
    end
    
    
    if distanceVector(u) == inf
        break
    end

    
    Q = setdiff(Q,u);
    lenQ = length(Q);
    
    for q = 1 : lenQ
        
        v = Q(q);
        
        if GraphDistanceMatrix(v,u) ~= inf
            
           % v is a neighbor of u
           alt = distanceVector(u) + GraphDistanceMatrix(v,u);
           
           if alt < distanceVector(v)
               distanceVector(v) = alt;
               %previousNodeID(v) = u;
           end
           
        end
    end % for
    
end % while


end % dijkstra()


