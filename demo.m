% Using 2D-CCA for blind source separation of 
% X = Wx*s + noise
% Y = Wy*s + noise
clear all 
close all 
addpath('Sparse2DCCA/')


n_rows_x = 10; 
n_cols_x = 10; 
n_rows_y = 10; 
n_cols_y = 10; 

n_samples = 2000;
k = 3;
s = zeros(k,n_samples);
for i = 1:k
    s(i,:) = i*sin(randi(1)*(1:n_samples)/(100*i) + randn(1));
end
% dur1      = 15; 
% dur2      = 20; 
% dur3      = 20;  
% fs        = 1;
% RT        = 1/fs;
% onsets1   = 00:30:n_samples-20;
% s(1,:) = get_stim(onsets1, dur1, n_samples);
% onsets2   = 20:45:n_samples-20;  
% s(2,:) = get_stim(onsets2, dur2, n_samples);


[x,W_x_cell] = create_data_matrix(n_rows_x,n_cols_x,s);
subplot(1,2,2)
[y,W_y_cell] = create_data_matrix(n_rows_y,n_cols_y,s);

figure
for i = 1:k
    subplot(k,2,2*(i-1)+1)
    imagesc(W_x_cell{i});
    subplot(k,2,2*i)
    imagesc(W_y_cell{i});
end


X = zeros(n_rows_x,n_cols_x,n_samples);
Y = zeros(n_rows_y,n_cols_y,n_samples);
for n = 1:n_samples
    X(:,:,n) = reshape(x(:,n),n_rows_x,n_cols_x);
    Y(:,:,n) = reshape(y(:,n),n_rows_y,n_cols_y);
end

% [Lx, Ly, Rx, Ry]=TWODCCA(X,Y,k+1,k+1,10);
options =struct('dim', k+1, 'LambdaU', 1,'LambdaV', 1,'EpsU', 1, 'EpsV', 1);
[Lx, Rx] = s2dcca_lowrank(X, Y, options);
[Ly, Ry] = s2dcca_lowrank(Y, X, options);

sx = zeros(size(Lx,2),n_samples);
sy = zeros(size(Lx,2),n_samples);
for n = 1:n_samples
    temp = Lx'*squeeze(X(:,:,n))*Rx;
    sx(:,n) = (diag(temp));
    temp = Ly'*squeeze(Y(:,:,n))*Ry;
    sy(:,n) = (diag(temp));
end

figure
subplot(311),plot(s')
subplot(312),plot((sx'))
subplot(313),plot((sy'))

figure
for i = 1:k
    subplot(k,2,2*(i-1)+1)
    imagesc(abs(reshape(x*sx(i,:)',n_rows_x,n_cols_x)));
    subplot(k,2,2*i)
    imagesc(abs(reshape(y*sy(i,:)',n_rows_y,n_cols_y)));
end

function [x,W_cell] = create_data_matrix(n_rows,n_cols,source)
% simulate data matrix using model:
% X = W*source + noise
[n_sources,n_samples] = size(source);
x = zeros(n_rows*n_cols,n_samples);
W_cell = cell(1,n_sources);
for k = 1:n_sources
    W = zeros(n_rows,n_cols);
    [sub_rows,sub_cols] = random_submatrix(n_rows,n_cols);
    W(sub_rows(1):sub_rows(2),sub_cols(1):sub_cols(2)) = 1;
    x = x + W(:)*source(k,:);
    W_cell{k} = W;
end
x = x + 0.1*randn(n_rows*n_cols,1);
x = bsxfun(@minus,x,mean(x,2));
end

function [row_idx,col_idx] = random_submatrix(n_rows,n_cols)
row_idx = sort(diag(randi(n_rows,2)));
col_idx = sort(diag(randi(n_cols,2)));
row_idx(2) = min(row_idx(1) + 1,row_idx(2));
col_idx(2) = min(col_idx(1) + 1,col_idx(2));
end