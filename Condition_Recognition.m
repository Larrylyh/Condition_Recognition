function RESULT = Condition_Recognition(train_Data, test_Data, train_ID, test_ID)
train_data=( train_Data - repmat(min(train_Data),size(train_Data,1),1) )./(...
    repmat(max(train_Data),size(train_Data,1),1) - repmat(min(train_Data),size(train_Data,1),1));
test_data = ( test_Data - repmat(min(test_Data),size(test_Data,1),1) )./(...
    repmat(max(test_Data),size(test_Data,1),1) - repmat(min(test_Data),size(test_Data,1),1));
trainGroup = train_ID;
testGroup = test_ID;

knn = fitcknn(train_data,trainGroup,'NumNeighbors',1,'Distance','euclidean');
classknn=predict(knn, test_data);
RESULT1 = percentage(testGroup,classknn);

multiSVM = fitcecoc(train_data,trainGroup);
classSVM=predict(multiSVM,test_data);
RESULT2 = percentage(testGroup,classSVM);
RESULT = [RESULT1 RESULT2];
end