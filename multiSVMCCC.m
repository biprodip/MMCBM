function [predicted_label, accuracy, modelprob]= multiSVMCCC(trainData,trainLabel,testData,testLabel)

addpath('libsvm/matlab');

%format given train data into libsvm format
libsvmwrite('datatrain.txt', trainLabel, sparse(trainData));
[trainLabel, trainData] = libsvmread('datatrain.txt');

%format given test data into libsvm format
libsvmwrite('datatest.txt', testLabel, sparse(testData));
[testLabel, testData] = libsvmread('datatest.txt');


% Parameter selection using 3-fold cross validation
% bestcv = 0;
% for log2c = -1:1:3,
%   for log2g = -4:1:2,
%     cmd = ['-q -c ', num2str(2^log2c), '-g', num2str(2^log2g),'-t 0'];
%     cv = get_cv_ac_1v1(trainLabel, trainData, cmd, 10);
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end


%bestParam = ['-q -c ', num2str(bestc), ' -g ', num2str(bestg),'-t 0'];

model = svmtrain(trainLabel,trainData,'-t 0 -b 1');
%bestParam
[predicted_label, accuracy, modelprob]=svmpredict(testLabel, testData, model,'-b 1');

end