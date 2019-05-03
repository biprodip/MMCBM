function y = recallCCC(M)
  y = diag(M) ./ sum(M,1)'; %1 is for sum of column elements
end