% Marvin Cao
% SID: 861214117
% 5/28/17
% CS-171: PS4
function findrules(D,smin,amin)
% Input: D(dataset), smin(minimum support), and amin(minimum confidence)
% find all association rules that meet the restrictions and writes all
% rules sorted by confidence

m = numexamples(D);
I = items(D);
% good enough support subsets go into S (in this case only those fo size 1
S = {};
for  i = 1:size(I,2)
    num = getcount(I(i),D) / m;
    % if enought support, add to S
    if num >= smin
        S = [S I(i)];
    end
end

% find all supported subsets and their corresponding support
Supported = {};
Support = {};
while (length(S) ~= 0)
    possible = aprioriGen(S); % generate possible sets of i+1 size
    Snext = {};
    for i=1:length(possible)
        % if enough support, add to Supported and Snext
        support_i = getcount(possible{i},D) / m;
        if support_i >= smin
            Snext = [Snext possible{i}];
            Support = [Support support_i];
        end
    end
    Supported = [Supported Snext];
    S = Snext;
end

% find all subsets with sufficient confidence from those with support
Rules = {};
for i = 1:size(Supported,2)
    subset = Supported{i};
    Rules = ruleGen(Rules,subset,[],D,amin);
end
% sort the rules
Rules = table2cell(sortrows(cell2table(Rules)));
% print to console
for i = 1:size(Rules,1)
    fprintf("%1.6f, %1.6f : %s\n",Rules{i,1},Rules{i,2}, rule2str(Rules{i,3},Rules{i,4},D));
end

%end of findrules

function C = aprioriGen(L)

C = {};
for j = 1:size(L,2)
    for k = (j+1):size(L,2)
        [diff,ia] = setdiff(L{j}, L{k}, 'sorted');
        if size(ia,1) == 1
            e = union(L{j}, L{k}, 'sorted');
            C = [C e];
        else
            break;
        end
    end
end

function Rules = ruleGen(Rules,X,Y,D,amin)
% takes in the previous set of rules, the subset partitions to work on, and confidence
% recursively, move elements from X to Y to check different rules

% if no X's to move to Y
if (size(X,2) == 1)
    return
end

% generate all rules made by moving something from X to Y
for i = 1:size(X,2)
    % generate the new rule (X and Y)
    tempY = sort([Y X(i)]);
    tempX = X;
    tempX(i) = [];
    % if already in rules, continue
    flag = 0;
    for j = 1:size(Rules,1)
        if isequal(tempX,Rules{j,3}) && isequal(tempY,Rules{j,4})
            flag = 1;
        end
    end
    if (flag)
        break;
    end
    
    % if sufficient confidence, add to Rules and recursively call
    confidence = getcount(union(tempX,tempY),D) / getcount(tempX,D);
    if confidence >= amin
        support = getcount(union(tempX,tempY),D) / numexamples(D);
        ruletemp = {confidence support tempX tempY};
        % add rule to Rules
        Rules = [Rules; ruletemp];
        % recursive call to next ones
        Rules = ruleGen(Rules,tempX,tempY,D,amin);
    end
end


