function [Lx,Ly,Rx,Ry,Xp,Yp] = Sparse_2DCCA_rank1(X,Y,options)
% This code implements sparse 2DCCA algorithm. 
% Inputs 
%        X : dataset of size px x qx x N
%        Y : dataset of size py x qy x N
% Options  :
%           : dimR    is the dimesion of the right canonical vectors
%           : dimL    is the dimesion of the right canonical vectors
%           : iter1   number of algorithm iterations for estimating
%                     left canonical vectors
%           : iter2   number of algorithm iterations for estimating right
%                     canonical vectors
%           : regwx   regularization parameter for X
%           : regwy   regularization parameter for Y
% Outputs   : Lx,Ly   left canonical vectors 
%           : Rx,Ry   right canonical vectors
%           % Xp      canonical variables obtained by projection of X
%           % Yp      canonical variables obtained by projection of Y
%Usage: 
%options=struct('iter1',5,'iter2',5,'dimR',5,'dimL',5,'regwx',1,'regwy',1,'projecL',1,'projecR',1);
% [Lxs,Lys,Rxs,Rys,~,~] = Sparse_2DCCA_rankoriginal(Dxx,Dyy,options);
%% Option parameters
[Mx,Nx,N] = size(X);
[My,Ny,N] = size(Y);

iter1 = options.iter1;
iter2 = options.iter2;
dr = options.dimR;
dl = options.dimL;
reg_wx = options.regwx;
reg_wy = options.regwy;

projetc_L = options.projecL; 
projetc_R = options.projecR; 

Xt = permute(X,[2,1,3]);  % Transpose des matices X
Yt = permute(Y,[2,1,3]);  % Transpose des matices Y
%Initialize Rx, Ry, Lx, Ly
Rx = eye(Nx,dr);
Ry = eye(Ny,dr);
Lx = eye(Mx,dl);
Ly = eye(My,dl);

for kk=1:iter1
    
    if (projetc_L==0)
        
        Lx = eye(Mx);
        Ly = eye(My);
        
    else
        Crxx = 10^(-6)*eye(Mx);
        Cryy = 10^(-6)*eye(My);
        Crxy = zeros(Mx,My);
        for jj=1:N
            Crxx = Crxx + (X(:,:,jj)*Rx*Rx'*X(:,:,jj)');
            Cryy = Cryy + (Y(:,:,jj)*Ry*Ry'*Y(:,:,jj)');
            Crxy = Crxy + (X(:,:,jj)*Rx*Ry'*Y(:,:,jj)');
        end
        Xr = reshape(X,Mx,Nx*N)*kron(eye(N),Rx);
        Yr = reshape(Y,My,Ny*N)*kron(eye(N),Ry);
        iCrxx = mldivide(Crxx,eye(Mx));
        iCryy = mldivide(Cryy,eye(My));
        
        clear Crxx Cryy;
        Px = Xr'*iCrxx*Xr;
        Py = Yr'*iCryy*Yr;
%         Kxy = Xr'*iCrxx*Crxy*iCryy*Yr;
        Kxy = Px*Py;
        clear Crxy;
        Lx = zeros(Mx,dl);
        Ly = zeros(My,dl);
        if (reg_wx==0 && reg_wy==0)

            [U,D,V] = svd(Kxy);
            Lx = iCrxx*Xr*U(:,1:dl)*D(1:dl,1:dl).^(1/2);
            Ly = iCryy*Yr*V(:,1:dl)*D(1:dl,1:dl).^(1/2);
            
            clear iCrxx iCryy;
        else
            clear iCrxx iCryy;
            for jj=1:dl 
                    [U,D,V] = svd(Kxy);
%                 [xLx,dummy] = eigs(Kxy,1);
%                 [yLy,dummy] = eigs(Kxy',1);
                    xLx = U(:,1);
                    yLy = V(:,1);

                for ii=1:iter2
                    Lxt = OMP_1D(Kxy*yLy,Xr',reg_wx);
                    xLx = Xr'*Lxt;
                    Lyt = OMP_1D(Kxy'*xLx,Yr',reg_wy);
                    yLy = Yr'*Lyt;

                end
                Xpr = Xr'*Lxt;
                Ypr = Yr'*Lyt;
                NoXpr = norm(Xpr,2);
                NoYpr = norm(Ypr,2);
                if NoXpr==0
                    ui = Xpr;
                else
                    ui = Xpr/NoXpr;
                end

                if NoYpr==0
                    vi = Ypr;
                else
                    vi = Ypr/NoYpr;
                end
                Kxy = Kxy - ui'*Kxy*vi*ui*vi';
                Lx(:,jj) = Lxt;
                Ly(:,jj) = Lyt;

            end

        end

        clear Xr Yr Kxy;
    end
    
%% ============ Calculate Rx and Ry =======================================

    if (projetc_R==0)
        Rx = eye(Nx);
        Ry = eye(Ny);
    else
        Clxx = 10^(-6)*eye(Nx);
        Clyy = 10^(-6)*eye(Ny);
        Clxy = zeros(Nx,Ny);
        for jj=1:N
            Clxx = Clxx + (X(:,:,jj)'*Lx*Lx'*X(:,:,jj));
            Clyy = Clyy + (Y(:,:,jj)'*Ly*Ly'*Y(:,:,jj));
            Clxy = Clxy + (X(:,:,jj)'*Lx*Ly'*Y(:,:,jj));
        end

        Xl = reshape(Xt,Nx,Mx*N)*kron(eye(N),Lx);
        Yl = reshape(Yt,Ny,My*N)*kron(eye(N),Ly);
        iClxx = mldivide(Clxx,eye(Nx));
        iClyy = mldivide(Clyy,eye(Ny));

        clear Clxx Clyy;
%         Qxy = Xl'*iClxx*Clxy*iClyy*Yl;
        Qxy = Px*Py;
        clear Clxy;
        Rx = zeros(Nx,dr);
        Ry = zeros(Ny,dr);
        if (reg_wx==0 && reg_wy==0)

            [U,D,V] = svd(Qxy);
            Rx = iClxx*Xl*U(:,1:dr)*D(1:dr,1:dr).^(1/2);
            Ry = iClyy*Yl*V(:,1:dr)*D(1:dr,1:dr).^(1/2);
            clear iClxx iClyy;
        else
            clear iClxx iClyy;
            for jj=1:dr   

                [U,D,V] = svd(Qxy);
                xRx = U(:,1);
                yRy = V(:,1);
%                 [xRx,dummy] = eigs(Qxy,1);
%                 [yRy,dummy] = eigs(Qxy',1);
                for ii=1:iter2
                    Rxt = OMP_1D(Qxy*yRy,Xl',reg_wx);
                    xRx = Xl'*Rxt;
                    Ryt = OMP_1D(Qxy'*xRx,Yl',reg_wy);
                    yRy = Yl'*Ryt;

                end
                Xpl = Xl'*Rxt;
                Ypl = Yl'*Ryt;
                NoXpl = norm(Xpl,2);
                NoYpl = norm(Ypl,2);

                if NoXpl==0
                    ui = Xpl;
                else
                    ui = Xpl/NoXpl;
                end

                if NoYpl==0
                    vi = Ypl;
                else
                    vi = Ypl/NoYpl;
                end
              Qxy = Qxy - ui'*Qxy*vi*ui*vi';
                Rx(:,jj) = Rxt;
                Ry(:,jj) = Ryt;

            end

        end

        clear Xl Yl Qxy;
        
    end


end

Xp = zeros(dl,dr,N);
Yp = zeros(dl,dr,N);

for ii=1:N
    
    Xp(:,:,ii) = Lx'*X(:,:,ii)*Rx;
    Yp(:,:,ii) = Ly'*Y(:,:,ii)*Ry;
end
% Xp = Lx'*X*kron(eye(N),Rx);
% Yp = Ly'*Y*kron(eye(N),Ry);

