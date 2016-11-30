function polenhan()

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
% track = '2ch';
% clear all variable for debugging
clear;
% default track
track = '6ch';

addpath ../utils;
addpath ../utils/ArrayToolbox;
% audio input path / segmented utterances
upath=['../../data/audio/16kHz/isolated_' track '_track/'];
% audio output path / enhanced utterances
epath=['../../data/audio/16kHz/z430_enhanced_super_' track '_track/'];
% path to continuous recordings
cpath='../../data/audio/16kHz/embedded/';
bpath='../../data/audio/16kHz/backgrounds/'; % path to noise backgrounds
% path to JSON annotations
apath=['../../data/annotations/'];

if strcmp(track, '6ch'),
  nchan = 5;
elseif strcmp(track, '2ch'),
  nchan = 2;
else
  error('This code is not suitable for singel channel data');
end

% Define hyper paramater
pow_thresh = -20;
% STFT window length
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

sets={'dt05'};
modes={'real', 'simu'};

for set_ind = 1:length(sets),
  set=sets{set_ind};
  for mode_ind = 1:length(modes),
    mode = modes{mode_ind};

    % Read annotations
    % get JSON list
    mat = json2mat([apath set '_' mode '.json']);

    snr = zeros(length(mat), 2);
    for utt_ind=1:length(mat),
      udir = [upath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
      edir = [epath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];

      if ~exist(edir, 'dir'),
        system(['mkdir -p ' edir]);
      end

      uname = [mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];

      % Load WAV file
      if strcmp(track, '6ch'),
        chanlist=[1 3:6];
      else
        wavlist = dir([udir uname '.CH*.wav']);
        for c = 1:nchan,
          wavname = wavlist(c).name;
          % get channel of wav files
          chanlist(c) = str2double(wavname(end-4));
        end
      end

      xsize = audioinfo([udir uname '.CH' int2str(chanlist(1)) '.wav']);
      % total number of audio samples in the file
      nsampl = xsize.TotalSamples;
      x = zeros(nsampl, nchan);

      % read wav files
      for c = 1:nchan,
        [x(:,c),fs] = audioread([udir uname '.CH' int2str(chanlist(c)) '.wav']);
      end

      % Check microphone failure
      if strcmp(track, '6ch'),
        xpow = sum(x.^2, 1);
        xpow = 10*log10(xpow/max(xpow));
        fail = (xpow <= pow_thresh);
      else
        fail = false(1, nchan);
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
          end
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
      stft_frame = stft_multi(x.', wlen);
      [nbin, nfram, ~] = size(stft_frame);
      % debugging
      % disp(size(stft_frame));
      [~, TDOA, srp, s] = localize(stft_frame, chanlist);
      % disp(srp);
      % compute noise coherence matrix at frequency-bin-k
      N=stft_multi(n.',wlen);
      Ncov=zeros(nchan,nchan,nbin);
      for f=1:nbin,
          for n=1:size(N,2),
              Ntf=permute(N(f,n,:),[3 1 2]);
              Ncov(:,:,f)=Ncov(:,:,f)+Ntf*Ntf';
              % Ncov(:,:,f)= sinc((2 * pi .* N * s) / c);

          end
          Ncov(:,:,f)=Ncov(:,:,f)/size(N,2);
      end
      % disp(I);
      % compute superdirective and steering vector
      Xspec = permute(mean(abs(stft_frame).^2, 2), [3 1 2]);
      mvdr = zeros(nbin, nfram);
      wng = 0.05;
      denomTemp = zeros(nsampl, 1);

      for f = 1:nbin,
        for t = 1:nfram,
          % steering vector
          Xtf = permute(stft_frame(f,t,:), [3 1 2]);
          d = sqrt(1/nchan)*exp(-2*1i*pi*(f-1)*TDOA(:,t));
          Ide = eye(size(Ncov(:, :, f)));
          % disp(size(diag(Xspec(~fail,f))));
          mvdr(f,t) = d(~fail)'/(Ncov(~fail,~fail,f)+wng*eye(size(diag(Xspec(~fail,f)))))*...
                Xtf(~fail)/(d(~fail)'/ ...
                (Ncov(~fail,~fail,f)+wng*eye(size(diag(Xspec(~fail,f)))))*d(~fail));
          wng = wng + 0.05;
          % denomTemp(t) = Ncov(~fail,~fail,f) * mvdr(:, f);
          % disp(denomTemp);

        end
      end
      output = istft_multi(mvdr, nsampl).';
      % disp(output);
      % disp(mvdr);
      % SNR calculation
      ENV_NUMBER=1;
      if strcmp(mat{utt_ind}.environment,'CAF'),
          ENV_NUMBER=2;
      elseif strcmp(mat{utt_ind}.environment,'PED'),
          ENV_NUMBER=3;
      elseif strcmp(mat{utt_ind}.environment,'STR'),
          ENV_NUMBER=4;
      end;
      % disp(size(output));
      snr(utt_ind,1) = 1;
      snr(utt_ind,2) = 10*log10(sum(output.^2) ./ sum(n.^2));
      % snr(utt_ind,2) = abs(output' * d(~fail)) ;
      disp(snr);
      disp('finish');
      % Write WAV file
      % y=y/max(abs(y));
      % audiowrite([edir uname '.wav'],y,fs);


    end
    csvwrite(['SNR_mvdr_' mode '.csv'],snr);
  end
end
