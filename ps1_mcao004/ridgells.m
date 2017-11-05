%Marvin Cao
%SID: 861214117
%4/13/17
%CS-171: PS1
function [w,b] = ridgells (X,Y,lambda)
% [w,b] = ridgells(X,Y,lambda)
%
% returns w and b which are the ridge regression fit to the data
% represented by X and Y using regularization constant lambda

%Preparation
% add a column of 1's to the end of X
X = [ones(size(X,1),1) X];
% get the dimension of the new X (m x n)
n = size(X,2);
%Xt * X is ultimately n x n since Xt is n x m and X is m x n and
% we need to add the lambda * identity mat
I = eye(n); % identity matrix of n x n
I(1,1) = 0;
% add the two together and take the inverse
A = (X'*X) + (lambda*I);
w = A\(X' * Y);
b = w(1);
w = w(2:size(w));