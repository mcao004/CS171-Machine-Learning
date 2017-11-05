% Marvin Cao
% SID: 861214117
% 5/28/17
% CS-171: PS4
function [Y,dt] = runq1()
% takes no parameters and returns vector of the predicted values on the
% testing data and the learned decision tree

all = importdata("banktrain.data");
testingX = importdata("banktestX.data");
% use last 35% for pruning
m = uint64(size(all,1) .* 0.65);
trainX = all(1:m,1:end-1);
trainY = all(1:m,end);
pruneX = all(m+1:end,1:end-1);
pruneY = all(m+1:end,end);

ftypes = [0 12 4 8 3 3 3 2 0 0 0 0 0 3 0 0 0 0 0];

% learn the decision tree
dt = learndt(trainX,trainY,ftypes,@misclassscore);
% prune
dt = prunedt(dt,pruneX,pruneY);
Y = predictdt(dt,testingX);
% draw the decision tree
% drawdt(dt);

function score = misclassscore(frac1)
score = min(frac1, 1-frac1);
% some kind of function to score a leaf and pass into
% learndt(X,Y,ftypes,@giniscore)
function score = giniscore(frac1)
score = frac1 .* (1-frac1);