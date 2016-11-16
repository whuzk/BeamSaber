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
c = 345.6,

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

      % steering vector
      % taken from localize function
      % remove zero frequency
      stft_frame = stft_frame(2:end,:,:);
      [nbin, nfram, nchan] = size(stft_frame);
      wlen = 2*nbin;
      % .' -> transpose matrix
      f = 16000 / wlen * (1:nbin).';

      % compute relative channel power
      if length(chanlist) > 2,
        xpow = shiftdim(sum(sum(abs(stft_frame).^2,2), 1));
        xpow = 10*log10(xpow / max(xpow)); % same on mic failure
      else
        xpow = zeros(1,2);
      end

      % define microphones position in centimeters
      xmic=[-10 0 10 -10 0 10]; % left to right axis
      ymic=[9.5 9.5 9.5 -9.5 -9.5 -9.5]; % bottom to top axis
      zmic=[0 -2 0 0 0 0]; % back to front axis
      xmic = xmic(chanlist);
      ymic = ymic(chanlist);
      zmic = zmic(chanlist);

      % define grid of possible speaker positions in centimeters
      xres = 46;
      xpos = linspace(-45, 45, xres);
      yres = 46;
      ypos = linspace(-45, 45, yres);
      zres = 4;
      zpos = linspace(15, 45, zres);
      ngrid = xres * yres * zres;

      % compute horizontal distances between grid points
      xvect = reshape(repmat(xpos.', [1 yres]), xres * yres, 1);
      yvect = reshape(repmat(ypos, [xres 1]), xres * yres, 1);
      pair_dist = sqrt((repmat(xvect, [1 xres * yres]) - repmat(xvect.', [xres * yres 1])).^2 + ...
                  (repmat(yvect, [1 xres * yres]) - repmat(yvect.', [xres * yres 1])).^2);

      % compute horizontal distance to the center
      center_dist = sqrt((xvect - mean(xpos)).^2 + (yvect - mean(ypos)).^2);

      % compute theoretical TDOAs between front pairs
      % speaker-to-microphone distances
      d_grid = zeros(nchan, xres, yres, zres);
      for c = 1:nchan,
        d_grid(c, :, :, :) = sqrt(repmat((xpos.' - xmic(c)).^2, [1 yres zres]) + ...
              repmat((ypos - ymic(c)).^2, [xres 1 zres]) + ...
              repmat((permute(zpos, [3 1 2]) - zmic(c)).^2, [xres yres 1]));
      end

      d_grid = reshape(d_grid, nchan, ngrid);
      pairs = [];
      for c = 1:nchan,
        % microphone pairs
        pairs = [pairs [c * ones(1, nchan - c); c + 1:nchan]];
      end
      % disp(pairs);
      npairs = size(pairs, 2);
      tau_grid = zeros(npairs, ngrid);
      for p = 1:pairs,
        c1 = pairs(1, p);
        c2 = pairs(2, p);
        tau_grid(p, :) = (d_grid(c2, :) - d_grid(c1, :)) / 343 / 100;
      end

      % compute SRP-PHAT pseudo-spectrum -> time delay between mic i-th to j-th
      srp = zeros(nfram, ngrid);
      for p = 1:npairs,
        c1 = pairs(1, p);
        c2 = pairs(2, p);
        d = sqrt((xmic(c1) - xmic(c2))^2 + (ymic(c1) - ymic(c2))^2 + (zmic(c1) - zmic(c2))^2);
        alpha = 10 * 343 / (d * 16000);
        lin_grid = linspace(min(tau_grid(p, :)), max(tau_grid(p, :)), 100);
        % GCC-PHAT pseudo-spectrum over a uniform interval
        lin_spec = zeros(nbin, nfram, 100);
        % discard channels with low power(mic failures)
        if (xpow(c1) > pow_thresh) && (xpow(c2) > pow_thresh),
          P = stft_frame(:, :, c1).* conj(stft_frame(:, :, c2));
          P = P./abs(P);
          for ind = 1:100,
            EXP = repmat(exp(-2 * 1i * pi * lin_grid(ind) * f), 1, nfram);
            lin_spec(:, :, ind) = ones(nbin, nfram) - tanh(alpha * real(sqrt(2 - 2 * real(P.*EXP))));
          end
        end
        lin_spec = shiftdim(sum(lin_spec, 1));
        tau_spec = zeros(nfram, ngrid);
        for t = 1:nfram,
          tau_spec(t, :) = interp1(lin_grid, lin_spec(t, :), tau_grid(p, :));
        end
        % sum over the microphone pairs
        srp = srp + tau_spec;
        % disp(srp);
      end

      % compute noise coherence matrix at frequency-bin-k
      Noise = stft_multi(n.', wlen);
      Ncoherence = zeros(nchan, nchan, nbin);
      for f = 1:bin,        
        Ncov(:, :, f) = sinc((2 * pi * freq * d_grid) / c);
      end


      % compute superdirective and steering vector


      disp('success');



    end
  end
end
