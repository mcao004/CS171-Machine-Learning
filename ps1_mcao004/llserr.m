%Marvin Cao
%SID: 861214117
%4/13/17
%CS-171: PS1
function ase = llserr (X,Y,w,b)
% ase = llserr(X,Y,w,b)
%
% Using w and b, predicts from each row in X the target, compares it to the
% actual target(in Y) and finds the squared error. Returns the averaged
% squared error over all such rows.

X = [ones(size(X,1),1) X];
w = [b; w];
X = (X*w) - Y;
ase = X' * X;
