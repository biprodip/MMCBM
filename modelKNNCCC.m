function result = modelKNNCCC(trainX,testX,trainY,k)

 %k= how many nearest neighbours for each sample classification 
 result=knnclassify(testX,trainX,trainY,k,'euclidean');

end