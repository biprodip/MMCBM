%clc;
%clear all;


res=zeros(1,1);  %result
avgP=zeros(1,1); %average precision
avgR=zeros(1,1); %average recall 
   
totCategory=6;          %total categories(backpack, helmet etc) %totAvilCategory=31; 
eachCatTrnSample=10;    %10  %each has samples
eachCatTstSample=10;    %each has samples

for trainDomain=2:2     %[A:1,D:2,W:3] %Set train and test dmain here
  for testDomain=3:3

%If train and test are from different domain than keep offset zero
trainOffset=0;  %train imges start from frame no 1+offset from directory #
testOffset=0;   %test images start from frame no 1+offset directory #

%trainDomain=3; %1=amazon, 2=dslr 3=web
%testDomain=3;  %1=amazon, 2=dslr 3=web

%totCategory=5;          %total categories(backpack, helmet etc) %totAvilCategory=31; 
%eachCatTrnSample=2;     %10  %each has samples
%eachCatTstSample=10;    %each has samples

domPartition=length(getDirectoryCCC())/3; %domain changing point
dirList=getDirectoryCCC();

TrainX=zeros(totCategory*eachCatTrnSample,4096); %each row is a sample
TrainY=zeros(totCategory*eachCatTrnSample,1);    %column vector

if trainDomain==1 %directory of amazon
  categorydir=1;
elseif trainDomain==2 %directory of dslr
  categorydir=1+domPartition; %11
elseif trainDomain==3  %directory of webcam
  categorydir=1+(domPartition*2); %21
end

totTrnSample=1;
for category=categorydir:(categorydir+totCategory-1)
  
  %category are helmet, backpack etc
  %   if (category==totCategory)||(category==totCategory+domPartition)
  %      category=category+(domPartition-totCategory+1);%goto new domain
  %      %y=de2bi(category,totCategory);
  %   end       
  
  if (category==10)||(category==20)||(category==30)
      clsIndx=category;
  else
      clsIndx=mod(category,10);  %categories starts at 1(dom1) or 11(dom2)  or 21(dom3)
  end
  
  for frameNo=1+trainOffset:eachCatTrnSample+trainOffset
     obj= load([dirList{category,:},sprintf('%04d.mat',frameNo)]);
     TrainX(totTrnSample,:)=obj.fc6'; %each row is a ptrn
     TrainY(totTrnSample,1)=clsIndx;  %corresponding clsIndx column is 1 others 0
     totTrnSample=totTrnSample+1;
  end  %sample
  
end   %category


TestX=zeros(totCategory*eachCatTstSample,4096);
%Load Test Samples
TestY=zeros(totCategory*eachCatTstSample,1);

if testDomain==1 %e.g. directory of amazon
  category=1;
elseif testDomain==2 %e.g. directory of dslr
  category=1+domPartition;
elseif testDomain==3  %e.g. directory of webcam
  category=1+(domPartition*2);
end

totTstSample=1;
for category=category:(category+totCategory-1)
  
  %category are helmet, backpack etc
  %   if (category==totCategory)||(category==totCategory+domPartition)
  %      category=category+(domPartition-totCategory+1);%goto new domain
  %      %y=de2bi(category,totCategory);
  %   end       
  
  if (category==10)||(category==20)||(category==30)
      clsIndx=category;
  else
      clsIndx=mod(category,10);  %categories starts at 1(dom1) or 11(dom2)  or 21(dom3)
  end
  
  for frameNo=1+testOffset:eachCatTstSample+testOffset
     obj= load([dirList{category,:},sprintf('%04d.mat',frameNo)]);
     TestX(totTstSample,:)=obj.fc6'; %each row is a ptrn
     TestY(totTstSample,1)=clsIndx;  %corresponding clsIndx column is 1 others 0
     totTstSample=totTstSample+1;
  end  %sample
  
end   %category

save('DecafTrain.mat');

%Linear SVM only approach
%[ACC(trainDomain,testDomain),confusionMat,avgPrecision(trainDomain,testDomain),avgRecall(trainDomain,testDomain)]=LSVM_Approach(TrainX,TestX,TrainY,TestY);

%MMCBM approach
[ACC(trainDomain,testDomain),confusionMat,avgPrecision(trainDomain,testDomain),avgRecall(trainDomain,testDomain)]=CCCBMCCC(TrainX,TestX,TrainY,TestY);

%Knn only approach
%[ACC(trainDomain,testDomain),confusionMat,avgPrecision(trainDomain,testDomain),avgRecall(trainDomain,testDomain)]=KNN_only(TrainX,TestX,TrainY,TestY);
res=res+ACC(trainDomain,testDomain);
avgP=avgP+avgPrecision(trainDomain,testDomain);
avgR=avgR+avgRecall(trainDomain,testDomain);  
  end
end


 
%  fresult=res./3
%  fpre=avgP./3
%  frec=avgR./3