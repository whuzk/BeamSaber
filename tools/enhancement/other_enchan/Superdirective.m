function Y = Superdirective(ftbin, Gcor, Ncor)
%SUPERDIRECTIVE Summary of this function goes here
%   Detailed explanation goes here
    [Nchan,Nbin,~] = size(ftbin);
    Df = zeros(Nchan, Nbin);
    disp('Df');
    disp(sizeof(Df));
    NcorInv = zeros(Nchan,Nchan,Nbin);
    
    
    f_center = f*fs/wlen;
    zeta = -1i*f_center*R*sin(theta)/c; 
    Df = [exp(zeta*cos(0*phi)); ...
                    exp(zeta*cos((2*pi/nchan)-1*phi)); ...
                    exp(zeta*cos((2*2*pi/nchan)-2*phi)); ...
                    exp(zeta*cos((2*3*pi/nchan)-3*phi)); ...
                    exp(zeta*cos((2*4*pi/nchan)-4*phi)); ...
                    exp(zeta*cos((2*5*pi/nchan)-5*phi)); ...
                    exp(zeta*cos((2*6*pi/nchan)-6*phi)); ...
                    exp(zeta*cos((2*7*pi/nchan)-7*phi))];
    
    for flp = 1:Nbin
        NcorInv(:,:,flp) = inv(Ncor(:,:,flp));
        [vv,dd] = eig(Gcor(:,:,flp));
        [~,ddind] = max(diag(dd));
        Df(:,flp) = vv(:,ddind);
    end
    %tdt : Nchan * 1 * Nbin
    tdt = sum(bsxfun(@times,NcorInv,permute(Df,[3,1,2])),2);
    Y = squeeze(sum(bsxfun(@times,conj(bsxfun(@rdivide, tdt, sum(bsxfun ...
        (@times,tdt,conj(permute(Df,[1,3,2]))),1))),permute(ftbin,[1,4,2,3])),1));
    
    mvdr = Df(:)'/...
                inv(Gcor(:,:,f)+wng*eye(size(diag(Xspec(:,f)))))* ...
                Xtf(:)/...
                (Df(:)'/inv(Gcor(:,:,f)+wng*eye(size(diag(Xspec(:,f)))))*...
                    Df(:));
   % squeeze eliminate singleton dimension
end

