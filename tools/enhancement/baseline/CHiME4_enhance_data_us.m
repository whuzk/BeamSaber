function CHiME4_enhance_data(track)

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

addpath ../../utils;
upath=['../../../../CHiME3/data/audio/16kHz/isolated_' track '_track/']; % path to segmented utterances
epath=['../../../../CHiME3/data/audio/16kHz/enhanced_' track 'r_track/']; % path to enhanced utterances
cpath='../../../../CHiME3/data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../../../../CHiME3/data/audio/16kHz/backgrounds/'; % path to noise backgrounds
apath=['../../../../CHiME3/data/annotations/'];
resultpath='../../../result/';
if strcmp(track,'6ch'),
    nchan=5;
elseif strcmp(track,'2ch'),
    nchan=2;
else
    error('This code is not suitable for single-channel data');
end

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length
regul=1e-3; % MVDR regularization factor
cmin=6400; % minimum context duration (400 ms)
cmax=12800; % maximum context duration (800 ms)

sets={'dt05'};
modes={'real' 'simu'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};

        % Read annotations
        % get JSON list
        mat=json2mat([apath set '_' mode '.json']);
        real_mat=json2mat([apath set '_real.json']);

        snr=zeros(length(mat),2);
        for utt_ind=1:length(mat),
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            edir=[epath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            if ~exist(edir,'dir'),
                system(['mkdir -p ' edir]);
            end
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];

            % Load WAV files
            if strcmp(track,'6ch'),
                chanlist=[1 3:6];
            else
                wavlist=dir([udir uname '.CH*.wav']);
                for c=1:nchan,
                    wavname=wavlist(c).name;
                    % get channel of wav files
                    chanlist(c)=str2double(wavname(end-4));
                end
            end
            xsize=audioinfo([udir uname '.CH' int2str(chanlist(1)) '.wav']);
            nsampl=xsize.TotalSamples; %Total number of audio samples in the file.
            x=zeros(nsampl,nchan);
            % read WAV files
            for c=1:nchan,
                [x(:,c),fs]=audioread([udir uname '.CH' int2str(chanlist(c)) '.wav']);
            end

            % Check microphone failure
            if strcmp(track,'6ch'),
                xpow=sum(x.^2,1);
                xpow=10*log10(xpow/max(xpow));
                fail=(xpow<=pow_thresh);
            else
                fail=false(1,nchan);
            end

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
                    n(:,c)=audioread([cpath cname '.CH' int2str(chanlist(c)) '.wav'],[cbeg cend]);
                    % disp([cpath cname '.CH' int2str(chanlist(c)) '.wav']);
                end
                % disp(nchan);
            elseif strcmp(set,'tr05'),
                cname=mat{utt_ind}.noise_wavfile;
                cbeg=max(round(mat{utt_ind}.noise_start*16000)-cmax,1);
                cend=max(round(mat{utt_ind}.noise_start*16000)-1,1);
                n=zeros(cend-cbeg+1,nchan);
                for c=1:nchan,
                    n(:,c)=audioread([bpath cname '.CH' int2str(chanlist(c)) '.wav'],[cbeg cend]);
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
                    n(:,c)=audioread([cpath cname '.CH' int2str(chanlist(c)) '.wav'],[cbeg cend]);
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
            [~,TDOA]=localize(X,chanlist);
            % display(TDOA);
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

            % SNR calculation
            ENV_NUMBER=1;
            if strcmp(mat{utt_ind}.environment,'CAF'),
                ENV_NUMBER=2;
            elseif strcmp(mat{utt_ind}.environment,'PED'),
                ENV_NUMBER=3;
            elseif strcmp(mat{utt_ind}.environment,'STR'),
                ENV_NUMBER=4;
            end;

            % Write WAV file
            y=y/max(abs(y));
            audiowrite([edir uname '.wav'],y,fs);
        end
        csvwrite([resultpath 'SNR_baseline_20data_' mode '.csv'],snr);

    end
end

return
