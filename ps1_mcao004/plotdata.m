%Marvin Cao
%SID: 861214117
%4/13/17
%CS-171: PS1
function plotdata(fname) 
% usage: [] = plotdata(fname)
%
% input a filename (as a string),and plots (in a single figure) each feature (column) in that data file 
% against the last column (the y value) as a scatter plot.

filename = fname;
delimiterIn = ' ';

% saves data into array A
A = importdata(fname,delimiterIn);
% get the dimensions of A
n = size(A,2);


y = A(:,n);
for k = 1:n-1
    x = A(:,k);
    res = subplot(ceil((n-1)/5), 5, k);
    scatter(res,x,y,4,'filled');
    xlabel(strcat("feature ", num2str(k)));
    ylabel("y");
end