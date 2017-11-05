%Marvin Cao
%SID: 861214117
%4/25/17
%CS-171: PS2
function [err,C] = knntest(TrainX, TrainY, TestX, TestY, k, lnorm)
%
% a stub
% your solution should report the total number of errors on the Test
% set using k-nearest neighbors with the supplied k and lnorm
% (lnorm=1 for Manhattan and 2 for Euclidean)
% It should also report C, the confusion matrix.  The i-j element of
% C is the fraction of the total examples who were labeled as class i
% and the true label was class j

minClass = min(TrainY);

crange = max(TrainY) - minClass;
%initialize the size of the confusion matrix
C = zeros(crange+1);

if lnorm == 1 %manhattan
    distmeas = 'cityblock';
else
    distmeas = 'euclidean';
end

%Mdl = fitcknn(TrainX, TrainY,'NumNeighbors',k,'Distance',distmeas);
for l = 1:size(TestY,1)
    predY = knnpredict(TrainX,TrainY,k,lnorm,TestX(l,:));
    C(predY - minClass + 1,TestY(l,1) - minClass + 1) = C(predY - minClass + 1,TestY(l,1) -minClass + 1) + 1;
end

% finish the confusion matrix
C = C./size(TestY,1);
% calculate error
err = 0;
for i = 1:crange+1
    for j = 1:crange+1
        if (i == j)
            continue
        end
        err = err + C(i,j);
    end
end

function predY = knnpredict(trainX,trainY, k, lnorm, x_attr)
% knn implementation for predicting x_attr
% 1 = manhattan dist, 2 = euclidean dist
[m,n] = size(trainX);

nearest = zeros(k,2); %holds index of nearest neighbors in trainX and the distance to x_attr
diff = zeros(m,1);
for i = 1:size(trainX,1)
    if (lnorm == 1)
        diff(i) = sum(abs(trainX(i,:) - x_attr));
    end
    if (lnorm == 2)
        diff(i) = sqrt(sum((trainX(i,:) - x_attr).^2));
    end
end

diff = [diff trainY];
sdiff = sortrows(diff);
minClass = min(trainY);
crange = max(trainY) - minClass;
classoccurrence = zeros(crange+1,1);
for f = 1:k
    classoccurrence(sdiff(f,end)+1) = classoccurrence(sdiff(f,end)+1) + 1;
end

[maxclass predY] = max(classoccurrence);
predY = predY-1;