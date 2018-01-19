function obj = initialize(obj, Ktensor_, C_, Theta_, Delta_, U_)
%% initialize
% 
% Public method of SIMDS class used to initialize SIMDS objects.
%
% NOTES
%   1. No input argument checking is being done!
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%

%% Initialize public data members
obj.Ktensor = Ktensor_;
obj.C = C_;
obj.Theta = Theta_;
obj.Delta = Delta_;
obj.U = U_;

%% Initialize private data members
[obj.N, dummy, obj.J] = size(obj.Ktensor);
obj.P = size(obj.C,1);
obj.SA = SIMDS.laplacian(obj.U);
obj.constant = 0.5 * sum(sum( obj.U .* obj.Delta .* obj.Delta ));
obj.UDelta = obj.U .* obj.Delta;
obj.Ip = eye(obj.P);
obj.In = eye(obj.N);
    
return % initialize()