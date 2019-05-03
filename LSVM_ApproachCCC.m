function [ACC,C,avgPrecision,avgRecall]=LSVM_ApproachCCC(TrainX,TestX,TrainY,TestY)
%clc 
%clear all
%set parameters in init Data Files and start
%initData;

%get model1 classification and probability output 
[predicted_label_1, accuracy1, modelprob1]= multiSVM(TrainX,TrainY,TestX,TestY);
save('modelout1.mat','modelprob1','predicted_label_1');

%finalResult
finalResult=predicted_label_1;

%Performance analysis
 
%Find Confusion martrix
  [C,order]=confusionmat(TestY,finalResult);
%Find ACC;
  ACC= sum(diag(C))/length(TestY);%(length(TestY)-length(find(TestY~=finalResult)))/length(TestY);
%Find Precision
  precision_scores=precision(C');
  avgPrecision=mean(precision_scores);
%Find Recall
  recall_scores=recall(C');
  avgRecall=mean(recall_scores);
end