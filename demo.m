% Demo for the paper "Face Image Classification by Pooling Raw Features",
% Pattern Recognition (PR), 2016.
% Written by Fumin Shen (fumin.shen AT gmail.com)


close all
clear

dataset = 'AR';
load(['testbed/' dataset])
fprintf([dataset ':\n']);

% Load  data
trainX = double(data_train.A');
trainY = double(data_train.label);
testX = double(data_test.Y');
testY = double(data_test.label);

rfSizes = [4 6 8];
% options for feature extraction
options.ReducedDim = 10;   
options.DIM = DIM;
options.pooling = 'max';
options.Pyramid = [1 1;2 2; 4 4; 6 6; 8 8; 10 10; 12 12; 15 15];


trainXC = [];testXC=[];
for ix_rf = 1:length(rfSizes)
    rfSize = rfSizes(ix_rf);
    options.rfSize = rfSize;
        
    numPatches = 50000;
    patches = zeros(numPatches, rfSize*rfSize);
    for i=1:numPatches
        if (mod(i,10000) == 0), fprintf('Extracting patches with rfSize %d: %d / %d\n', rfSize, i, numPatches); end
        r = random('unid', DIM(1) - rfSize + 1);
        c = random('unid', DIM(2) - rfSize + 1);
        patch = reshape(trainX(random('unid', size(trainX,1)),:), DIM);
        patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);
        patches(i,:) = patch(:)';
    end
    
    % normalize for contrast
    patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));
    % PCA
    options.eigvector = PCA(patches,options);
 
    fprintf('extracting features for training data... \n');
    tem = extract_feature(trainX, options);
    trainXC = [trainXC, tem];

    fprintf('extracting features for testing data... \n');
    tem = extract_feature(testX, options);
    testXC = [testXC, tem];
end

% standardize data
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);

trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];

testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% classification
[W] = RRC(trainXCs, trainY(:), 0.005); % you can also use other classifiers like standard SVMs.
[val,labels] = max(testXCs*W, [], 2);
acc = sum(labels == testY) / length(testY);
fprintf('test accuracy %f%%\n', 100 * acc);