function [Xp, R2, alpha, reflection] = procrustes(X, Y, use_dilation)
%% PROCRUSTES
%
% Performs Procrustes analysis on two datasets and returns one of the sets
% appropriately rotated/reflected/dilated/translated to match the other one.
%
% SYNTAX
%   [Xp, R2, alpha, reflection] = procrustes(X, Y, use_dilation);
%
% INPUTS
%   X: NxD data matrix containing N d_dimensional samples of the first set 
% in its rows.
%   Y: NxD data matrix containing N d_dimensional samples of the second set 
% in its rows.
%   use_dilation: integer scalar. If ~=0, optimize w.r.t. dilation 'alpha'; 
% otherwise, do not.
%
% OUTPUTS
%   Xp: NxD transformed data matrix of first set. It is rotated/reflected/
% dilated/translated to optimally match the set Y.
%   R2: scalar in [0,1]. Provides the Procrustes statistic R^2. Its
% square root is known as (normalized) Procrustes distance between the two 
% sets.
%   alpha: real scalar; optimal dilation (scaling) value. If use_dilation=0,
% then alpha=1.0 will be returned.
%   reflexion: integer scalar; if =1, dataset X is related to dataset Y via
% one or more reflections; otherwise, =0.
%
% NOTES
%   1. No input argument checking is being done!
%   2. R2=0 indicates a perfect match.
%
% COPYRIGHT
%   Mahlagha Sedghi
%	ayuz1qiz@gmail.com
%

%% Display help text, if no arguments are provided
if nargin == 0
    eval('help procrustes')
    return
end

%% Initializations
[N,D] = size(X);
alpha = 1.0;

%% Compute sample means
meanX_row = mean(X);
meanY_row = mean(Y);

%% Compute centered data
Xc = X - repmat(meanX_row,N,1);
Yc = Y - repmat(meanY_row,N,1);

%% Compute optimal rotation/reflection matrix H
Z = Yc' * Xc;
[U, S, V] = svd(Z);
H = V*U';
reflection = (sign(det(H)) == -1);

%% Compute optimal dilation
tmp1 = trace(Xc' * Xc);
tmp2 = trace(H * Z);
if use_dilation ~= 0
    alpha = tmp2 / tmp1;
end

%% Calculate Procrustes statistic
tmp3 = trace(Yc * Yc');
R2 = tmp3 + alpha^2 * tmp1 - 2.0 * alpha * tmp2;
R2 = R2 / tmp3; % normalize

%% Produce tranformed data of first set to best match second dataset
Xp = alpha * Xc * H + repmat(meanY_row,N,1);

%% Return to MATLAB
return




