%Marvin Cao
%SID: 861214117
%5/12/17
%CS-171: PS3
function [W1,W2] = trainneuralnet(X,Y,nhidden,lambda)
% [W1,W2] = trainneuralnet(X,Y,nhidden,lambda);
% 
eta = 0.01;
W1 = randn(nhidden,size(X,2)+1);
W2 = randn(1,nhidden+1);
% load in and set up trainX as the training x pits and trainY as the Y pts
gridX = getgridpts(X);
% train classifier
it = 0;
thousands = 0;
while 1
    [y_hat, z1] = forwardprop(W1,W2,X); % get all predicted values of y
    deltaW1 = zeros(size(W1));
    deltaW2 = zeros(size(W2));
    for k = 1:size(y_hat)
        [tempdeltaW1,tempdeltaW2] = backwardprop(W1,W2,y_hat(k),Y(k),z1(:,k),X(k,:),lambda);
        deltaW1 = deltaW1 + tempdeltaW1;
        deltaW2 = deltaW2 + tempdeltaW2;
    end
    deltaW1 = (deltaW1 ./ size(X,1)) + (2*lambda).*W1;
    deltaW2 = (deltaW2 ./ size(X,1)) + (2*lambda).*W2;
    
    % if done
    gradmax = max([max(max(abs(deltaW1))) max(max(abs(deltaW2)))]);
    if (gradmax <= 1e-4)
        break
    end
    
    W1 = W1 - eta .* deltaW1;
    W2 = W2 - eta .* deltaW2;
    
    if mod(it,10000) == 0
        % display loss function's value
        [gridY,gridZ] = forwardprop(W1,W2,gridX);
        loss = evaluate(X,Y,y_hat,lambda,W1,W2);
        %W1
        %disp(deltaW1);
        %W2
        %disp(deltaW2);
        [lambda nhidden (thousands./1000) loss gradmax]
        %disp(evaluate(X,Y,gridY,lambda,W1,W2));
        plotdecision(X,Y,gridX,gridY,.5);
        % plot surface
        drawnow;
        it = 0;
        thousands = thousands + 10;
    end
    it = it+1;
end

[gridY,gridZ] = forwardprop(W1,W2,gridX);
% use classifier to predict labels for gridX, call this vector gridY
plotdecision(X,Y,gridX,gridY);

function loss = evaluate(X,Y,predY,lambda,W1,W2)
loss = (-Y .* log(predY)) - ((1-Y).*log(ones(size(predY))-predY));
loss = sum(sum(loss)) / size(X,1);
sweight = lambda .* (sum(sum(W1.*W1)) + sum(sum(W2.*W2)));
loss = loss + sweight;

function [y_hat,z1] = forwardprop(W1,W2,X)
% [y_hat,z] = forwardprop(W1,W2,X);
% Given an mxn matrix X, calculates a predicted y_hat of 1xm and z1 of
% (nhidden+1) x m
X = X'; % need the instances to be column vectors, not row vectors
X = [ones(1,size(X,2)); X]; % add offset
a = W1*X;
z1 = 1 ./(1+exp(-a)); % sigmoid function
z1 = [ones(1,size(z1,2)); z1]; % offset
a_last = W2*z1; % column vector of all the predicted y's
y_hat =  (1 ./(1+exp(-a_last)))';



function [deltaW1,deltaW2] = backwardprop(W1,W2,y_hat,y,z1,X,lambda)
% [delta1,delta2] = backwardprop(W1,W2,y_hat,z1,eta)
% only use individual y_hat, y and single vector X
X = [ones(1,size(X,1)); X']; % instance to a column vector
delta = y_hat - y;
% generate other delta(s)
delta1 = (z1.*(ones(size(z1))-z1)) .* (W2' * delta);
% return update values
deltaW2 = delta * z1';
deltaW1 = (delta1(2:end)) * X';