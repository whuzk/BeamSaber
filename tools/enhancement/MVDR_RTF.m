function Y = MVDR_RTF(ftbin, Gcor, Ncor, refMic)

[Nchan,Nbin,~] = size(ftbin);
u0 = zeros(Nchan,1);
u0(refMic) = 1;
ng = zeros(Nchan,Nchan,Nbin);
lambda = zeros(Nbin,1);
    
for flp = 1:Nbin
    ng(:,:,flp) = Ncor(:,:,flp) \ Gcor(:,:,flp);
    lambda(flp) = trace(ng(:,:,flp));
end
Y = bsxfun(@rdivide, squeeze(sum(bsxfun(@times, conj(sum(bsxfun(@times, ng, u0.'), 2)), permute(ftbin,[1,4,2,3])),1)),lambda);