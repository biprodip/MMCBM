function [ACC,C,avgPrecision,avgRecall]=CCCBMCCC(TrainX,TestX,TrainY,TestY)
clc 
%clear all

%get model1 classification and probability output 
[predicted_label_1, accuracy1, modelprob1]= multiSVMCCC(TrainX,TrainY,TestX,TestY);
save('modelout1.mat','modelprob1','predicted_label_1');

%get clustering output
[clusterOut,centroid,intraSum,DtoC]=clusteringCCC(TestX,TrainY);
save('clusterOut.mat','clusterOut','centroid','intraSum','DtoC');


%Local weight clustering for each sample's hardness calculation 
%It loads cluster and classification model output and computes weight
[lweight,gweight,en_prob,en_pred]=LWCCCC();
save('weights.mat','lweight','gweight','en_prob','en_pred');

%Find weak weight samples & Classify using KNN
weakIndices=find(lweight<=.5);
weakSamples=TestX(weakIndices,:);
k=3;
weakResult=modelKNNCCC(TrainX,weakSamples,TrainY,k);

%finalResult
finalResult=en_pred;
finalResult(weakIndices)=weakResult;


%Performance analysis
 
%Find Confusion martrix
  [C,order]=confusionmat(TestY,finalResult);
%Find ACC;
  ACC= sum(diag(C))/length(TestY);%(length(TestY)-length(find(TestY~=finalResult)))/length(TestY);
%Find Precision
  precision_scores=precisionCCC(C');
  avgPrecision=mean(precision_scores);
%Find Recall
  recall_scores=recallCCC(C');
  avgRecall=mean(recall_scores);
%Find AUC
 % labels=TestY;
 % scores=finalResult;
 % posclass=3;
 % [X,Y,T,AUC] = perfcurve(labels,scores,posclass);
 % plot(X,Y);
end