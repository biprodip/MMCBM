clc
clear all;

valDomain=3;  %1=amazon, 2=dslr 3=web

totCategory=5;          %total categories(backpack, helmet etc) %totAvilCategory=31; 
eachCatValSample=4;     %10  %each has samples

%If train and test are from different domain than keep offset zero
valOffset=0; %train imges start from frame no 1+offset from directory #

domPartition=length(getValDirectory())/3; %domain changing point
dirList=getValDirectory();

valX=zeros(totCategory*eachCatValSample,4096); %each row is a sample
valY=zeros(totCategory*eachCatValSample,1);    %column vector

if valDomain==1 %directory of amazon
  categorydir=1;
elseif valDomain==2 %directory of dslr
  categorydir=1+domPartition; %7
elseif valDomain==3  %directory of webcam
  categorydir=1+(domPartition*2); %13
end

totValSample=1;
for category=categorydir:(categorydir+totCategory-1)
    
      
  if (category==6)||(category==12)||(category==12)
          clsIndx=6;
  else
      clsIndx=mod(category,6);  %categories starts at 1(dom1) or 11(dom2)  or 21(dom3)
  end
  
  for frameNo=1+valOffset:eachCatValSample+valOffset
     obj= load([dirList{category,:},sprintf('%04d.mat',frameNo)]);
     ValX(totValSample,:)=obj.fc6'; %each row is a ptrn
     ValY(totValSample,1)=clsIndx;  %corresponding clsIndx column is 1 others 0
     totValSample=totValSample+1;
  end  %sample
  
end   %category

totValSample=totValSample-1;
save('DecafVal.mat');

%load 'DecafVal.mat';

minRow=min(ValX);
minVal=min(minRow);
[r,c]=size(ValX);
scale=repmat(-minVal,r,c);
sclValX=ValX+scale;

noOfCluster=size(unique(ValY),1);

[clusterOutVal,centroidVal,intraSumVal,DtoCVal]= kmeans(sclValX,noOfCluster,'distance','sqEuclidean','emptyaction','singleton');

%Find cluster purity
noClass=noOfCluster;
maxVal=zeros(noOfCluster,1);
for group=1:noOfCluster
   %get a cluster group
     group_indices=find(clusterOutVal==group);
     
     for class=1:noClass
       %get class1 samples(base result) 
       class_indices=find(ValY==class);
       %match with cluster groups to find correspondence
       match(class)=length(intersect(group_indices,class_indices));
       
       if match(class)>maxVal(group)
           maxVal(group)=match(class);
           %max_class_indices=class_indices; %maximum matched indices
           %indx=class;
       end
       
     end          
     %tmpClusterLabels(group_indices)=[indx];
     %check it
     %groupPurity(indx)=maxVal/length(union(group_indices,max_class_indices));
end

purity=sum(maxVal)/totValSample
      
save('clusterOutVal.mat','clusterOutVal','centroidVal','intraSumVal','DtoCVal');