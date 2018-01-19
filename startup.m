%% STARTUP
%
% Script to be automatically run by MATLAB upon launch, if it is located
% in the user's MATLAB directory. It adds the nececessary SIMDS bundle
% directories to the MATLAB path. The paths shown below must be modified to
% reflect the correct paths of the system in use.
%

 
%% Setup paths for SIMDS code and utilities
addpath C:\Users\mahlagha\Documents\MATLAB\SIMDS
addpath C:\Users\mahlagha\Documents\MATLAB\utils

%% Notify user 
disp('startup.m just ran!')