function ACC = KNN_onlyCCC(TrainX,TestX,TrainY,TestY)
%clc 
%clear all
%set parameters in init Data Files and start
%initData;


%Find weak weight samples & Classify using KNN
k=3;
Result=modelKNN(TrainX,TestX,TrainY,k);
finalResult=Result;


%Performance analysis
 
%Find ACC;
  ACC= (length(TestY)-length(find(TestY~=finalResult)))/length(TestY);
%Find Confusion martrix
  [C,order]=confusionmat(TestY,finalResult);
%Find Precision
  precision_scores=precision(C);
  avgPrecision=mean(precision_scores);
%Find Recall
  recall_scores=recall(C);
  avgRecall=mean(recall_scores);
%Find AUC
 % labels=TestY;
 % scores=finalResult;
 % posclass=3;
 % [X,Y,T,AUC] = perfcurve(labels,scores,posclass);
 % plot(X,Y);
end