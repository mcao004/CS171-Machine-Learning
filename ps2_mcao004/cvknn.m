%Marvin Cao
%SID: 861214117
%4/25/17
%CS-171: PS2
function [k,lnorm] = cvknn(Xtrain,Ytrain,Xvalid,Yvalid,maxk)
% 
% a starting shell
% your solution should find the best k (from 1 to maxk, skipping even values)
% and lnorm (1 = Manhattan distance, 2 = Euclidean distance) combination
% for k-nearest neighbor using the supplied training and validation sets
%
% In doing so, it should generate a plot (do *not* call "figure" -- the
% calling function will set up for the right figure window to be active).
% The plot should be as described and illustrated in the problem set. 
rangek = 1:2:maxk;

manerr = zeros(1,size(rangek,2));
% for each k using manhattan dist
for k = 1:size(rangek,2)
    for i = 1:size(Yvalid)
        predY = knnpredict(Xtrain,Ytrain,rangek(k),1,Xvalid(i,:));
        
        if predY ~= Yvalid(i)
            manerr(k) = manerr(k)+1;
        end
    end
end
manerr = manerr./size(Yvalid,1);

eucerr = zeros(1,size(rangek,2));
% now the same for euclidean dist
for k = 1:size(rangek,2)
    for i = 1:size(Yvalid)
        predY = knnpredict(Xtrain,Ytrain,rangek(k),2,Xvalid(i,:));
        
        if predY ~= Yvalid(i)
            eucerr(k) = eucerr(k)+1;
        end
    end
    eucerr(k) = eucerr(k)./size(Yvalid,1);
end

%for our return values
if min(manerr) > min(eucerr)
    lnorm = 2;%Euclidean has the smallest error
    [M,I] = min(eucerr);%I has index of the smallest error in eucerr
    k = rangek(I);
else
    lnorm = 1;%Manhattan
    [M,I] = min(manerr);
    k = rangek(I);
end

plot(rangek,manerr);
hold on;
plot(rangek,eucerr);
hold off;
legend('Manhattan','Euclidean');
ylabel('error rate');
xlabel('k');

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
