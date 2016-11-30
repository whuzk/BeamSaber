function CHiME3_enhance_data

% CHIME3_ENHANCE_DATA Enhances noisy datasets for the 3rd CHiME Challenge
%
% CHiME3_enhance_data
%
% If you use this software in a publication, please cite:
%
% Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
% third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
% task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
% and Understanding Workshop (ASRU), 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015 University of Sheffield (Jon Barker, Ricard Marxer)
%                Inria (Emmanuel Vincent)
%                Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath ../utils;
upath='../../data/audio/16kHz/isolated/'; % path to segmented utterances
epath='../../data/audio/16kHz/enhanced/'; % path to enhanced utterances
cpath='../../data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../../data/audio/16kHz/backgrounds/'; % path to noise backgrounds
apath='../../data/annotations/'; % path to JSON annotations
nchan=6;

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length
regul=1e-3; % MVDR regularization factor
cmin=6400; % minimum context duration (400 ms)
cmax=12800; % maximum context duration (800 ms)

sets={'tr05' 'dt05' 'et05'};
modes={'real' 'simu'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};
        
        % Read annotations
        mat=json2mat([apath set '_' mode '.json']);
        real_mat=json2mat([apath set '_real.json']);
            
        for utt_ind=1:length(mat),
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            edir=[epath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            if ~exist(edir,'dir'),
                system(['mkdir -p ' edir]);
            end
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            
            % Load WAV files
            xsize=wavread([udir uname '.CH1.wav'],'size');
            nsampl=xsize(1);
            x=zeros(nsampl,nchan);
            for c=1:nchan,
                [x(:,c),fs]=wavread([udir uname '.CH' int2str(c) '.wav']);
            end
            
            % Check microphone failure
            xpow=sum(x.^2,1);
            xpow=10*log10(xpow/max(xpow));
            fail=(xpow<=pow_thresh);
            
            % Load context (up to 5 s immediately preceding the utterance)
            if strcmp(mode,'real'),
                cname=mat{utt_ind}.wavfile;
                cbeg=max(round(mat{utt_ind}.start*16000)-cmax,1);
                cend=max(round(mat{utt_ind}.start*16000)-1,1);
                for utt_ind_over=1:length(mat),
                    cend_over=round(mat{utt_ind_over}.end*16000);
                    if strcmp(mat{utt_ind_over}.wavfile,cname) && (cend_over >= cbeg) && (cend_over < cend),
                        cbeg=cend_over+1;
                    end
                end
                cbeg=min(cbeg,cend-cmin);
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=wavread([cpath cname '.CH' int2str(c) '.wav'],[cbeg cend]);
                end
            elseif strcmp(set,'tr05'),
                cname=mat{utt_ind}.noise_wavfile;
                cbeg=max(round(mat{utt_ind}.noise_start*16000)-cmax,1);
                cend=max(round(mat{utt_ind}.noise_start*16000)-1,1);
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=wavread([bpath cname '.CH' int2str(c) '.wav'],[cbeg cend]);
                end
            else
                cname=mat{utt_ind}.noise_wavfile;
                cbeg=max(round(mat{utt_ind}.noise_start*16000)-cmax,1);
                cend=max(round(mat{utt_ind}.noise_start*16000)-1,1);
                for utt_ind_over=1:length(real_mat),
                    cend_over=round(real_mat{utt_ind_over}.end*16000);
                    if strcmp(mat{utt_ind_over}.wavfile,cname) && (cend_over >= cbeg) && (cend_over < cend),
                        cbeg=cend_over+1;
                    end
                end
                cbeg=min(cbeg,cend-cmin);
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=wavread([cpath cname '.CH' int2str(c) '.wav'],[cbeg cend]);
                end
            end
    
            % STFT
            X = stft_multi(x.',wlen);
            [nbin,nfram,~] = size(X);
            
            % Compute noise covariance matrix
            N=stft_multi(n.',wlen);
            Ncov=zeros(nchan,nchan,nbin);
            for f=1:nbin,
                for n=1:size(N,2),
                    Ntf=permute(N(f,n,:),[3 1 2]);
                    Ncov(:,:,f)=Ncov(:,:,f)+Ntf*Ntf';
                end
                Ncov(:,:,f)=Ncov(:,:,f)/size(N,2);
            end
            
            % Localize and track the speaker
            [~,TDOA]=localize(X);
            
            % MVDR beamforming
            Xspec=permute(mean(abs(X).^2,2),[3 1 2]);
            Y=zeros(nbin,nfram);
            for f=1:nbin,
                for t=1:nfram,
                    Xtf=permute(X(f,t,:),[3 1 2]);
                    Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen*fs*TDOA(:,t)); % steering vector
                    Y(f,t)=Df(~fail)'/(Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f)))*Xtf(~fail)/(Df(~fail)'/(Ncov(~fail,~fail,f)+regul*diag(Xspec(~fail,f)))*Df(~fail));
                end
            end
            y=istft_multi(Y,nsampl).';
            
            % Write WAV file
            y=y/max(abs(y));
            wavwrite(y,fs,[edir uname '.wav']);
        end
    end
end

return
