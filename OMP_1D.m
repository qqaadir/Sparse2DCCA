function [c] = OMP_1D(x,D,L);
%=============================================
% Sparse coding of a single signal based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).
%       x - the signal to represent
%       L - the max. number of coefficients for each signal.
% output arguments: 
%       c - sparse coefficient vector.
%=============================================
n = length(x);
x = x(:);
[n,K] = size(D);
indx = zeros(L,1);
residual = x;

for j = 1:L, 
    proj = D'*residual; 
    [maxVal,pos] = max(abs(proj));
    indx(j) = pos;
    s = pinv(D(:,indx(1:j)))*x;
    residual = x - D(:,indx(1:j))*s;
    if sum(residual.^2) < 1e-6
        break;
    end,
end;
    temp = zeros(K,1);
    temp(indx(1:j)) = s;
    c = sparse(temp);
return;


end

