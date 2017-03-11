function mvdr()

addpath ../../utils;
% Define hyper-parameters
chanlist=[1:8]; % number of microphone array
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length
regul=1e-3; % MVDR regularization factor
cmin=6400; % minimum context duration (400 ms)
cmax=12800; % maximum context duration (800 ms)

% load original file
[x, fs] = wavread('audio/2m_pub_new16khz.wav');

% load noise file
[n, fs] = wavread('audio/noise_1258.wav');

% STFT
[nsampl, nchan] = size(x);
X = stft_multi(x.',wlen);
[nbin,nfram,~] = size(X);

% Compute noise covariance matrix
N=stft_multi(n.',wlen);
Ncov=zeros(nchan,nchan,nbin);
disp(size(X));
for f=1:nbin,
	for n=1:size(N,2),
		Ntf=permute(N(f,n,:),[3 1 2]);
		Ncov(:,:,f)=Ncov(:,:,f)+Ntf*Ntf';
	end
	Ncov(:,:,f)=Ncov(:,:,f)/size(N,2);
end
% Localize and track the speaker
[~,TDOA]=localize(X,chanlist);
% display(TDOA);
% MVDR beamforming
Xspec=permute(mean(abs(X).^2,2),[3 1 2]);
Y=zeros(nbin,nfram);
for f=1:nbin,
	for t=1:nfram,
		Xtf=permute(X(f,t,:),[3 1 2]);
		Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen*fs*TDOA(:,t)); % steering vector
		Y(f,t)=Df(:)'/(Ncov(:,:,f)+regul*diag(Xspec(:,f)))*Xtf(:)/(Df(:)'/(Ncov(:,:,f)+regul*diag(Xspec(:,f)))*Df(:));
	end
end
y=istft_multi(Y,nsampl).';

% Write WAV file
y=y/max(abs(y));
audiowrite('2m_mvdr.wav',y,fs);
end
% csvwrite([resultpath 'SNR_baseline_20data_' mode '.csv'],snr);


