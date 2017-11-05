%Marvin Cao
%SID: 861214117
%4/25/17
%CS-171: PS2
function runq1
% tries logistic regression on the data in the file phishing.dat
% Treat the entire dataset as training data
% should load the data and try both linear and quadratic models (by augmenting the feature space).
% Try different regularization strengths

training = importdata("phishing.dat");
trainX = [ones(size(training,1),1) training(:,1:size(training,2)-1)]; %features of urls augmented with 1 at beginning
trainY = training(:,size(training,2));

% we already have w = learnlogreg(X,Y,lambda) to perform logistic
% regression on the data, we just need to try linear and quadratic models
% with different lambdas

% create the quadratic model
quad = trainX;
for k = 2:size(trainX,2)
   for l = k:size(trainX,2)
      quad = [quad trainX(:,k).*trainX(:,l)];
   end
end

% lambdas
%ls = logspace(-3,5,20);
ls = [.001 .01 .1 1 10 100 1000 10000];
%linearw = zeros(size(trainX,1), 0);
%quadw = zeros(size(quad,1), size(quad,2));

best = inf;
bestlambda = 1;
lin = 0; %type of best is 1 or 2 for linear and quadratic respectively

for l=ls
    linearw = learnlogreg(trainX,trainY,l);
    fx = trainX*linearw;
    err = 0;
    for i = 1:size(fx,1)
        if fx(i)*trainY(i) < 0
            err = err+1;
        end
    end
    err = err./size(fx,1);
    
    if (lin == 0)
        best = err;
        lin = 1;
        bestlambda = l;
    end
    if (best > err) 
        %better error
        best = err;
        lin = 1;
        bestlambda = l;
    end
    
    quadw = learnlogreg(quad,trainY,l);
    fx = quad*quadw;
    err = 0;
    for i = 1:size(fx,1)
        if fx(i)*trainY(i) < 0
            err = err+1;
        end
    end
    err = err./size(fx,1);
    
    if (best > err) 
        %better error
        best = err;
        lin = 2;
        bestlambda = l;
    end
end

disp("lowest error");
disp(best);
if (lin == 1)
    disp("linear");
else
    disp("quadratic");
end
disp("best lambda");
disp(bestlambda);