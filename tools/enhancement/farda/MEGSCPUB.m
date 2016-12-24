%% GSC 
% Yfbf = MVDR Baseline
% Lower path yg dari farda
function MEGSCPUB()

% CHIME4_ENHANCE_DATA Enhances noisy datasets for the 4th CHiME Challenge
% based on MVDR beamforming
% 
% Note: This code is identical to the CHiME-3 baseline, except that only
% the channels corresponding to each track are used. MVDR is known to work
% poorly on real data due to the fact that it does not handle microphone
% mismatches, microphone failures, early echoes, and reverberation. This
% code is not intended to be run as such (the official CHiME-4 baseline
% based on BeamformIt provides much better results) but to provide a set of
% Matlab tools from which more advanced beamforming or source separation
% techniques can be developed.
%
% CHiME4_enhance_data(track)
%
% Inputs:
% track: '2ch' or '6ch'
%
% If you use this software in a publication, please cite:
%
% Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
% third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
% task and baselines, in Proc. IEEE 2015 Automatic Speech Recognition
% and Understanding Workshop (ASRU), 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015-2016 University of Sheffield (Jon Barker, Ricard Marxer)
%                     Inria (Emmanuel Vincent)
%                     Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fbf='MEGSCPUB';
track='6ch';
addpath enhan;
addpath utils;

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length
regul=1e-3; % MVDR regularization factor
cmin=6400; % minimum context duration (400 ms)
cmax=12800; % maximum context duration (800 ms)
Nsource = 8;
EMITERNUM = 20;

 [x,fs]=audioread('2m_pub_new.wav');
 [nsampl,nchan]=size(x);
 display(nchan);
    
            % STFT
            X = stft_multi(x.',wlen);
            [nbin,nfram,~] = size(X);
            
          
            % Localize and track the speaker
            %[~,TDOA]=localize(X,[1:8]);
            
            Xbin = permute(X,[3 1 2]);
             % correlation of X
            XX = bsxfun(@times, permute(Xbin,[1,4,2,3]), conj(permute(Xbin,[4,1,2,3])));
	    Xcor = mean(XX, 4);
            % Ncor = mean(XX(:,:,:,[1:10,Nframe-10:Nframe]),4);
            % Load context (up to 5 s immediately preceding the utterance)
            % noise = read_context(nchan, c_ind, Path, mode, mat, real_mat, uttInd, cmax, cmin); 
            
            softmask = cGaussMask(Xbin,Nsource,XX,EMITERNUM);
            Ncor = bsxfun(@rdivide, mean(bsxfun(@times, XX, permute(softmask(:,:,2),[3,4,1,2])),4),permute(mean(softmask(:,:,2),2), [2,3,1]));                       
            Gcor = Xcor - Ncor;
            %Xmask = bsxfun(@times, Xbin, permute(softmask(:,:,1),[3,1,2]));
            display(size(X));
             % for GSC fixed beamformer
            
%             targetY = squeeze(yFbf(refMic,:,:));
            
            % MVDR beamforming
%             Xspec=permute(mean(abs(X).^2,2),[3 1 2]);
% 			Yfbf=zeros(nbin,nfram);
% %             Y=zeros(nbin,nfram);
%             for f=1:nbin,
%                 for t=1:nfram,
% %                     Xtf=permute(X(f,t,:),[3 1 2]);
%                     Xtf = Xmask(:,f,t);
%                     Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen*fs*TDOA(:,t)); % steering vector
%                     Yfbf(f,t)=Df(~fail)'/(Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f)))*Xtf(~fail)/(Df(~fail)'/(Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f)))*Df(~fail));
%                 end
%             end
            Yfbf = MVDR_EV(Xbin, Gcor, Ncor);
            
% 			Yfbf = bsxfun(@times, Yfbf, permute(softmask(:,:,1),[3,1,2]));  
			B = zeros(nchan-1,nchan,nbin);

            for lp = 1:nchan-1
                B(lp,lp,:)=1;
                B(lp,lp+1,:)=-1;
            end
    
            z = BM(B, permute(X,[3 1 2]));
            R = zeros(nchan-1, nbin);
            %[RR, ~] = AF(targetY, z, ftbin, R);
            %[~, Y] = AF(targetY, z, ftbin, RR);
            [~, Y] = AF(Yfbf, z, permute(X,[3 1 2]), R); 
			
            y=istft_multi(Y,nsampl).';
            
            % Write WAV file
            y=y/max(abs(y));
            audiowrite(['2m_pub_new_' fbf '.wav'],y,fs);
return
end

function [R, e] = AF(b, z, ftbin, R)
% z is the output of block matrix, (nchan-1) * nbin * nframe 
% b is the output of the fixed beamformer, nbin * nframe
% E is the ftbin of the microphone nchan * nbin * nframe

% R is the output of the microphone: (nchan-1) * nbin
% e is the output of AD, is the desired signal: nbin * nframe
[~, Nbin, Nframe] = size(ftbin);
mu = 0.005;
alpha = 0.1;
e = zeros(Nbin, Nframe);
%R = zeros(Nchan-1, Nbin);
P = zeros(Nbin, 1);
    for tlp = 1:Nframe
        e(:,tlp) = b(:,tlp) - permute(sum(conj(R).*z(:,:,tlp),1),[2,1]); % e = b(:,tlp) - R' * z;
        P = alpha * P + (1 - alpha) * permute(sum(conj(ftbin(:,:,tlp).*ftbin(:,:,tlp)),1),[2,1]);  %P = alpha * P + (1 - alpha) * sum(ftbin);
        %P = alpha * P + (1 - alpha) * permute(sum(conj(z(:,:,tlp).*z(:,:,tlp)),1),[2,1]);  %P = alpha * P + (1 - alpha) * sum(ftbin);
        R = R + mu * bsxfun(@times, z(:,:,tlp), permute(conj(e(:,tlp))./(P+eps), [2,1]));    % R = R + mu * e * z;
    end
end

function z = BM(B, ftbin)
% B is (nchan-1) * nchan * nbin
% ftbin is nchan * nbin * nframe
% z is (nchan -1) * nbin * nframe
z = squeeze(sum(bsxfun(@times, conj(B),  permute(ftbin, [4,1,2,3])),2));
end
