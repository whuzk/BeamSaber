function Ncov = ncov_diagonal(ftbin, noise)
  [Nchan,Nbin,~] = size(ftbin);
  Ncov = zeros(Nchan, Nchan, Nbin);
  for f = 1:Nbin,
    for n = 1:size(noise, 2),
      Ntf = permute(noise(f,n,:), [3 1 2]);
      Ncov(:,:,f) = Ncov(:,:,f)+Ntf+Ntf';
    end
    Ncov(:,:,f) = Ncov(:,:,f)/size(noise, 2)
end


function Y = mvdr_diagonal(ftbin, noiseCovariance)

  [Nchan, Nbin, ~] = size(ftbin);
  steeringVector = zeros(Nchan, Nbin);
  


end
