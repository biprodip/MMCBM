function y = precisionCCC(M)
  y = diag(M) ./ sum(M,2); %2 is for sum of row elements
end

