
track = '6ch';

addpath ../../utils;
resultpath = ['../../../result/'];
nchan = 8;
% Define hyper paramater
pow_thresh = -20;
% STFT window length
% wlen = 1024;
% wlen for steering vector estimation
wlen = 1024;
% weight given to the prior that the speaker's horizontal position is close to the center
center_factor = 0.05;
% weight given to the transition probabilities
smoothing_factor = 3;
% speed of sound recordings
c = 345.6;
% minimum context duration (400 ms)
cmin = 6400;
% maximum context duration (800 ms)
cmax = 12800;

wng = 0.05;
count = 0;
chanlist=[1:8];
[x,fs] = audioread('audio/2m_pub_new16khz.wav');
[noise, fs] = audioread('audio/noise_1258.wav');

[nsampl, nchan] = size(x);
signal = stft_multi(x.', wlen);

% disp(size(signal));
[nbin, nfram, ~] = size(signal);
% debugging
% disp(size(stft_frame));
[~, TDOA] = localize(signal, chanlist);
disp(size(TDOA));
N=stft_multi(noise.',wlen);
disp(size(N));

noise_per = permute(N, [3 2 1]);
% disp(d);
steering_vector = zeros(nbin, nfram);
for f= 1:nbin,
    for t = 1:nfram,
        a = exp(-2*1i*pi.*noise_per(:, t, f));
        disp(size(a));
        steering_vector(f,t) = abs(a)*TDOA(:, t);
    end
end
disp(size(steering_vector));
% compute noise coherence matrix at frequency-bin-k
N=stft_multi(noise.',wlen);
Ncov=zeros(nchan,nchan,nbin);
for f=1:nbin,
    for n=1:size(N,2),
        Ntf=permute(N(f,n,:),[3 1 2]);
        Ncov(:,:,f)=Ncov(:,:,f)+Ntf*Ntf';
    end
    Ncov(:,:,f)=Ncov(:,:,f)/size(N,2);
end

  % compute superdirective and steering vector
Xspec = permute(mean(abs(signal).^2, 2), [3 1 2]);
mvdr = zeros(nbin, nfram);
wng = 0.05;
fail = false(1, nchan);
for f = 1:nbin,
    disp('nbin');
    for t = 1:nfram,
%       disp('loop every frame');
      % steering vector
      Xtf = permute(signal(f,t,:), [3 1 2]);
       Df = sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen*fs.*TDOA(:,t)); % steering vector
      % superdirective beamforming
      % you can use bsxfun to do matrix calculation even different size
      noiser = 0;
      if mvdr(f,t) > -10,
        mvdr(f,t) = Df(~fail)'/...
                    inv(Ncov(~fail,~fail,f)+wng*eye(size(diag(Xspec(~fail,f)))))* ...
                    Xtf(~fail)/...
                    (Df(~fail)'/inv(Ncov(~fail,~fail,f)+wng*eye(size(diag(Xspec(~fail,f)))))*...
                    Df(~fail));
        % it's still confused with the noise gain calculation,
        % is wng the output of mvdr?
        % noisegain = (abs(mvdr(f,t)' * Df(~fail)).^2) / (mvdr(f,t)' * mvdr(f,t));
        % noiser = sum(noisegain);
        wng = wng + 0.05;
        if mvdr(f,t) <= -10,
          break;
        end
      end
    end
end


% break;
output = istft_multi(mvdr, nsampl).';
% Write WAV file
output=output/max(abs(output));

audiowrite('2m_super-test_.wav',output,fs);





