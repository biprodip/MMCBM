function [local_weight,groupWeight,w_prob,CLU,CLA]=weightingCCC(CLU,prob)

%  weighting: compute local model weights according to the similarity
%             between the model and the clustering structure. (works for
%             binary classification problems)
%  Input: 
%       CLU- a column vector where each element represents the cluster
%            membership (0 or 1).
%       prob- a n by 2 matrix where n is the number of test examples, 
%             containing the predictions made by a classification model.  
%             prob(i,1)- the probability of i-th example belonging to class
%             0; prob(i,2)- the probability of i-th example belonging to
%             class 1.
%  Output:           
%       weight- a colum vector where each element represents the model
%               weight at the test example.
%       w_prob- the model's predictions weighted by its local weights
%
%==========================================================================
%
%
% An example:
%
%
%       directory='results'; % the result folder
% 
%       clusterfile=[directory,'/cluster'];
%       CLU=load(clusterfile); % the clustering results
%
%       num=length(CLU);
%       W=zeros(num,3);
%       en_prob=zeros(num,2); % initialization
%       
%       for i=1:3 % compute model weights
%        pfile=[directory,'/predict',num2str(i)];
%        prob=load(pfile);
%        [weight,w_prob]=weighting(CLU,prob);
%        en_prob=en_prob+w_prob;
%        W(:,i)=weight;
%       end
% 
%       en_prob=en_prob./repmat(sum(en_prob,2),1,2); % the ensemble output
%
%       enfile=[directory,'/ensemble'];
%       csvwrite(enfile,en_prob); %output the ensemble predictions
%
%       wfile=[directory,'/weight_matrix'];
%       csvwrite(wfile,W); %output the weights


num=length(CLU);
CLA=zeros(length(CLU),1);
[val,CLA]=max(prob');           %id=find(prob(:,2)>0.5); %select maximum max(prob) 
CLA=CLA';
%CLA(indx)=1;                   %update accordingly

tmpClusterLabels=zeros(length(CLU),1); %to store corrected clustering results

noClass=length(unique(CLU));
%indx=0;

for group=1:noClass
   %get a cluster group
     group_indices=find(CLU==group);
     maxVal=0;
     for class=1:noClass
       %get samples of class1(base classifier's result) 
       class_indices=find(CLA==class);
       %match with cluster groups to find correspondence
       match(class)=length(intersect(group_indices,class_indices));
       
       if match(class)>maxVal
           maxVal=match(class);              %maximum match between clstr samples and clssification
           max_class_indices=class_indices;  %maximum matched indices
           indx=class;
       end
       
     end          
     tmpClusterLabels(group_indices)=[indx];
     %check it
     groupWeight(indx)=maxVal/length(union(group_indices,max_class_indices));
end

CLU=tmpClusterLabels;

for i=1:num                 %each sample's weight
    index=[1:i-1,i+1:num];  %excluding i
    ulabel=CLU(i);          %cluster output
    alabel=CLA(i);          %classifier output
    
    id1=find(CLU(index)==ulabel);
    id2=find(CLA(index)==alabel);
	
    %the percentage of common neighbors
    local_weight(i,1)=length(intersect(id1,id2))/length(union(id1,id2));
end


r=repmat(local_weight,1,noClass); %num X num
w_prob=prob.*r;
end