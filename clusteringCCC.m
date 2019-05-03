function [clusterOut,centroid,intraSum,DtoC]= clusteringCCC(testData,trainLabel)

minRow=min(testData);
minVal=min(minRow);
[r,c]=size(testData);
scale=repmat(-minVal,r,c);
sclTestX=testData+scale;

noOfCluster=size(unique(trainLabel),1);

%%Spectral clustering
%noEigVec=9;
%eigenEnergy=.20;
%clusterOut=Jordan_Weiss(sclTestX,noOfCluster,noEigVec,eigenEnergy);
% opts = statset('Display','final');

%%KMeans clustering
[clusterOut,centroid,intraSum,DtoC]= kmeans(sclTestX,noOfCluster,'distance','sqEuclidean','emptyaction','singleton','start','cluster');

%clusterOut=clusterOut-1;%clst no. 1 or 2 now 0 or 1
end