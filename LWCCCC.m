      function [lweight,gweight,en_prob,en_pred]= LWCCCC()
       load clusterOut.mat;
       load modelout1.mat;
       
       noModels=1;                     %individual base models
       CLU=clusterOut;
       num=length(CLU);
       noCluster=length(unique(CLU));
       
       W=zeros(num,noModels);          %as there are two models so (n x noModels) matrix
       en_prob=zeros(num,noCluster);   %initi ensembled prob(two classes so, numx2)
       
       for modelno=1:noModels          % compute model weights
        model_prob=eval(sprintf('modelprob%d',modelno));
        [lweight,gweight,w_prob,CLU,CLA]=weightingCCC(CLU,model_prob);
        en_prob=en_prob+w_prob;
        W(:,modelno)=lweight;
       end
 
       en_prob=en_prob./repmat(sum(en_prob,2),1,noCluster); % the ensemble output(0-1 range)
       
       en_pred=zeros(length(en_prob),1);
       [val,en_pred]=max(en_prob');
       en_pred=en_pred';
       %id=find(en_prob(:,2)>0.5);
       %en_pred(id)=1;
       
       %en_pred=en_pred+1;   %0/1 to 1/2

       
       
       
       
%        vec_res=ind2vec(en_pred'); % 1 at corresponding class in each column(ptrn)
%        vec_tar=ind2vec(tstClass'); % 1 at corresponding class in each column(ptrn)
% 
%        RC=figure()
%        plotconfusion(vec_tar,vec_res);
%        
       
%        plotroc(vec_tar,vec_res);
%        [X,Y,T,AUC] = perfcurve(tstClass,en_pred,'2');
%        plot(X,Y,'r-');
%        saveas(RC,'Result/full_en_12.jpg','jpg');
%        


%        enfile=[directory,'/ensemble'];
%        csvwrite(enfile,en_prob); %output the ensemble predictions
% 
%        wfile=[directory,'/weight_matrix'];
%        csvwrite(wfile,W); %output the weights
      end