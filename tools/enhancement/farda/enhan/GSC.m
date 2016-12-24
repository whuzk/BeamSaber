function [Y, N] = GSC(ftbin, targetY)
% GSC frond-end. 
% just a simple one, may not well-implemented
    [Nchan, Nbin, ~] = size(ftbin);
    B = zeros(Nchan-1,Nchan,Nbin);

    for lp = 1:Nchan-1
        B(lp,lp,:)=1;
        B(lp,lp+1,:)=-1;
    end
        
%         Df = zeros(Nchan, Nbin);
%         for flp = 1:Nbin
%             [vv,dd] = eig(Gcor(:,:,flp));
%             [~,ddind] = max(diag(dd));
%             Df(:,flp) = vv(:,ddind);
%         end
%         H = -bsxfun(@rdivide, Df, Df(1,:));
%         H = conj(H(2:end,:));
%         B(:,1,:) = permute(H, [1,3,2]);
%         B(:,2:end,:) = repmat(eye(Nchan-1),[1,1,Nbin]);

    z = BM(B, ftbin);
    R = zeros(Nchan-1, Nbin);
    %[RR, ~] = AF(targetY, z, ftbin, R);
    %[~, Y] = AF(targetY, z, ftbin, RR);
    [~, Y, N] = AF(targetY, z, ftbin, R);    
end

function [R, e, n] = AF(b, z, ftbin, R)
% z is the output of block matrix, (nchan-1) * nbin * nframe 
% b is the output of the fixed beamformer, nbin * nframe
% E is the ftbin of the microphone nchan * nbin * nframe

% R is the output of the microphone: (nchan-1) * nbin
% e is the output of AD, is the desired signal: nbin * nframe
[~, Nbin, Nframe] = size(ftbin);
mu = 0.005;
alpha = 0.1;
e = zeros(Nbin, Nframe);
n = zeros(Nbin, Nframe);
%R = zeros(Nchan-1, Nbin);
P = zeros(Nbin, 1);
    for tlp = 1:Nframe
        %n(:,tlp) = permute(sum(conj(R).*z(:,:,tlp),1),[2,1]);
        e(:,tlp) = b(:,tlp) - permute(sum(conj(R).*z(:,:,tlp),1),[2,1]); % e = b(:,tlp) - R' * z;
        P = alpha * P + (1 - alpha) * permute(sum(conj(ftbin(:,:,tlp).*ftbin(:,:,tlp)),1),[2,1]);  %P = alpha * P + (1 - alpha) * sum(ftbin);
        %P = alpha * P + (1 - alpha) * permute(sum(conj(z(:,:,tlp).*z(:,:,tlp)),1),[2,1]);  %P = alpha * P + (1 - alpha) * sum(ftbin);
        R = R + mu * bsxfun(@times, z(:,:,tlp), permute(conj(e(:,tlp))./(P+eps), [2,1]));    % R = R + mu * e * z;
        n = e-b ;    
    end
	
end

function z = BM(B, ftbin)
% B is (nchan-1) * nchan * nbin
% ftbin is nchan * nbin * nframe
% z is (nchan -1) * nbin * nframe
z = squeeze(sum(bsxfun(@times, conj(B),  permute(ftbin, [4,1,2,3])),2));
end
    