function [Alpha, Beta] = s2dcca_lowrank(X, Y, options)

% -------------------------------------------------------%
% Sparse 2DCCA via low rank implementation

% Description: This code implements sparse 2DCCA via low rank approximation 
% paper (SPL, 2012)by jingjie Yan et.al.

% Inputs :  
%          X: First 3D data matrix 
%          Y: Second 3D data matrix
%    options: 
%           : dim is the dimesion of the canonical vectors
%           : LambdaU is the regularization parameter
%           : LambdaV is the regularization parameter
%           : EpsU    is the covergence tolerance
%           : EpsV    is the covergence tolerance 
% Outputs:  
%      Alpha: Left canonical vector
%      Beta : Right canonical vectors
% Usage : options =struct('dim', 5, 'LambdaU', 1,'LambdaV', 1,'EpsU', 0.1, 'EpsV', 0.1);
%                  [Alpha, Beta] = s2dcca_lowrank(Dx, Dy, options);
%% -------------------------------------------------------%
%% Parameters
Dimension_CCA = options.dim;
LambdaU       = options.LambdaU;
LambdaV       = options.LambdaV;
EpsU          = options.EpsU;
EpsV          = options.EpsV; 
%% Data centering
[mx, nx, N] = size(X);
[my, ny, N] = size(Y);

Xc=zeros(mx,nx,N);
Yc=zeros(my,ny,N);

for i=1:N
    Xc(:,:,i)=bsxfun(@minus, X(:,:,i), mean(X, 3));
    Yc(:,:,i)=bsxfun(@minus, Y(:,:,i), mean(Y, 3));
end
%% Computation of covariance matrices
Gab=10^(-6)*eye(nx,ny);
Ga=10^(-6)*eye(nx,nx);
Gb=10^(-6)*eye(ny,ny);

for num=1:N
    Gab=Gab+(X(:,:,num)'*Y(:,:,num));
    Ga=Ga+(X(:,:,num)'*X(:,:,num));
    Gb=Gb+(Y(:,:,num)'*Y(:,:,num));
end 

Gab=Gab/N;
Ga=Ga/N;
Gb=Gb/N;

%% Computation of 'K'.
K=((power(Ga,-1/2))*Gab*(power(Gb,-1/2)));
%% SVD of 'K'.
[U, D, V]=svd(K);

% 't' is the Number of Desired Canonical Variates.
t=Dimension_CCA;

% Canonical Correlation Variables.
uFinal=zeros(nx,t);Alpha=zeros(nx,t);
vFinal=zeros(ny,t);Beta=zeros(ny,t);

% Computation of 't' Canonical Correlation Variables.
for num=1:t
    flag=0;
    
    %Step 1 Initialise 'u' and 'v'.
    u_Old=D(num,num)*U(:,num);
    v_Old=V(:,num);
  
    %Repeat Till 'u' and 'v' Converge.
   while(flag==0)
    %Step 2 Update 'V'
    temp1=K'*u_Old;
    temp2=0.5*LambdaV*ones(ny,1);
    temp3=abs(temp1)-temp2;
    v_New=times(sign(temp1), max(temp3,0));
    if(norm(v_New,2)~=0)
        v_New=v_New/(norm(v_New,2));
    end 
    %Step 3 Update 'U'
    temp1=K*v_Old;
    temp2=0.5*LambdaU*ones(nx,1);
    temp3=abs(temp1)-temp2;
    u_New=times(sign(temp1), max(temp3,0));
    % Error Check.
    errorU=norm((u_New-u_Old),2);
    errorV=norm((v_New-v_Old),2);
    % Check for Convergence.
    if((errorU<EpsU)&&(errorV<EpsV))
        flag=1;
    end
    u_Old=u_New;
    v_Old=v_New;
    
  end 
  %Step 5
  if(norm(u_New,2)~=0)
    uFinal(:,num)=u_New/(norm(u_New,2));
  end 
  vFinal (:,num)=v_New;
  
  %Compute FInal Projection Directions from 'u' and 'v'.
  Alpha(:,num)=((power(Ga,-1/2))*uFinal(:,num));
  Beta(:,num)=((power(Gb, -1/2))*vFinal(:,num));
  
  %Step 6 Deflate 'K', to be used in the computation of next Canonical
  %correlation variables.
  K=K-(uFinal(:,num)'*K*vFinal (:,num)*uFinal(:,num)*vFinal (:,num)');
end 
