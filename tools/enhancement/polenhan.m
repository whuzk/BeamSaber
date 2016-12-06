function polenhan()

% track = '2ch';
% clear all variable for debugging
clear;
% default track
track = '6ch';

addpath ../utils;
% addpath ../utils/ArrayToolbox;
% audio input path / segmented utterances
% change data to remote directory to safe the space add "../CHiME4/CHiME3/"
% upath=['../../data/audio/16kHz/isolated_' track '_track/']; % path to segmented utterances
upath=['../../../CHiME3/data/audio/16kHz/isolated_' track '_track/'];
% audio output path / enhanced utterances
epath=['../../../CHiME3/data/audio/16kHz/z430_enhanced_super_' track '_track/'];
% path to continuous recordings
cpath='../../../CHiME3/data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../../data/audio/16kHz/backgrounds/'; % path to noise backgrounds
% cpath='../../../CHiME3/data/audio/16kHz/embedded/';
% bpath='../../../CHiME3/data/audio/16kHz/backgrounds/'; % path to noise backgrounds
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
sets={'dt05'};
modes={'real', 'simu'};

for set_ind = 1:length(sets),
  set=sets{set_ind};
  for mode_ind = 1:length(modes),
    count = count + 1;
    if count > 5,
      break;
    end

    mode = modes{mode_ind};

    % Read annotations
    % get JSON list
    mat = json2mat([apath set '_' mode '.json']);
    real_mat=json2mat([apath set '_real.json']);

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
        % disp([udir uname '.CH' int2str(chanlist(c)) '.wav']);
        % disp(size(x.'));
      end

      % Check microphone failure
      if strcmp(track, '6ch'),
        xpow = sum(x.^2, 1);
        xpow = 10*log10(xpow/max(xpow));
        fail = (xpow <= pow_thresh);
      else
        fail = false(1, nchan);
      end

      % % Load context (up to 5 s immediately preceding the utterance)
      % if strcmp(mode,'real'),
      %     cname=mat{utt_ind}.wavfile;
      %     cbeg=max(round(mat{utt_ind}.start*16000)-cmax,1);
      %     cend=max(round(mat{utt_ind}.start*16000)-1,1);
      %     for utt_ind_over=1:length(mat),
      %         cend_over=round(mat{utt_ind_over}.end*16000);
      %         if strcmp(mat{utt_ind_over}.wavfile,cname) && (cend_over >= cbeg) && (cend_over < cend),
      %             cbeg=cend_over+1;
      %         end
      %     end
      %     cbeg=min(cbeg,cend-cmin);
      %     n=zeros(cend-cbeg+1,nchan);
      %     for c=1:nchan,
      %         n(:,c)=audioread([cpath cname '.CH' int2str(chanlist(c)) '.wav'],[cbeg cend]);
      %     end
      % elseif strcmp(set,'tr05'),
      %     cname=mat{utt_ind}.noise_wavfile;
      %     cbeg=max(round(mat{utt_ind}.noise_start*16000)-cmax,1);
      %     cend=max(round(mat{utt_ind}.noise_start*16000)-1,1);
      %     n=zeros(cend-cbeg+1,nchan);
      %     for c=1:nchan,
      %         n(:,c)=audioread([bpath cname '.CH' int2str(chanlist(c)) '.wav'],[cbeg cend]);
      %     end
      % else
      %     cname=mat{utt_ind}.noise_wavfile;
      %     cbeg=max(round(mat{utt_ind}.noise_start*16000)-cmax,1);
      %     cend=max(round(mat{utt_ind}.noise_start*16000)-1,1);
      %     for utt_ind_over=1:length(real_mat),
      %         cend_over=round(real_mat{utt_ind_over}.end*16000);
      %         if strcmp(mat{utt_ind_over}.wavfile,cname) && (cend_over >= cbeg) && (cend_over < cend),
      %             cbeg=cend_over+1;
      %         end
      %     end
      %     cbeg=min(cbeg,cend-cmin);
      %     n=zeros(cend-cbeg+1,nchan);
      %     for c=1:nchan,
      %         n(:,c)=audioread([cpath cname '.CH' int2str(chanlist(c)) '.wav'],[cbeg cend]);
      %     end
      % end
      % STFT
      % disp(size(n));
      signal = stft_multi(x.', wlen);
      % disp(size(signal));
      [nbin, nfram, ~] = size(signal);
      % debugging
      % disp(size(stft_frame));
      [~, TDOA, srp, s] = localize(signal, chanlist);

      % define microphone positions in cm
      xmic = [-10 0 10 -10 0 10];
      ymic = [9.5 9.5 9.5 -9.5 -9.5 -9.5];
      zmic = [0 -2 0 0 0 0];

      d = zeros(length(xmic), length(xmic));
      % disp(d);
      for i = 1:length(xmic),
          se = 0;
          for j = 1:length(xmic),
              d(i,j) = sqrt((xmic(i) - xmic(j))^2 + ...
                            (ymic(i) - ymic(j))^2 + ...
                            (zmic(i) - zmic(j))^2);
          end
      end
      % compute noise coherence matrix at frequency-bin-k
      % N=stft_multi(n.',wlen);
      Ncov=zeros(nchan,nchan,nbin);
      i = 1;
      j = 1;
      for f = 1:nbin,
        Ncov(:,:,f) = sinc(2*pi*fs*d(i,j)/c);
        if j >=6,
          j = 1;
          i = i+1;
        end
        if i >=6,
          i = 1;
        end
        j = j + 1;
      end

      % disp('should break here');
      % for f=1:nbin,
      %     for n=1:size(N,2),
      %         Ntf=permute(N(f,n,:),[3 1 2]);
      %         Ncov(:,:,f)=Ncov(:,:,f)+Ntf*Ntf';
      %         % d = eig(Ncov);
      %         % Ncov(:,:,f)= sinc((2 * pi * n * s) / c);
      %
      %     end
      %     Ncov(:,:,f)=Ncov(:,:,f)/size(N,2);
      % end
      % disp(Ncov);
      % break;
      % disp(I);
      % compute superdirective and steering vector
      Xspec = permute(mean(abs(signal).^2, 2), [3 1 2]);
      mvdr = zeros(nbin, nfram);
      wng = 0.05;

      for f = 1:nbin,
        for t = 1:nfram,
          % steering vector
          Xtf = permute(signal(f,t,:), [3 1 2]);
          Df = sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen*fs*TDOA(:,t)); % steering vector
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
      ENV_NUMBER=1;
      if strcmp(mat{utt_ind}.environment,'CAF'),
          ENV_NUMBER=2;
      elseif strcmp(mat{utt_ind}.environment,'PED'),
          ENV_NUMBER=3;
      elseif strcmp(mat{utt_ind}.environment,'STR'),
          ENV_NUMBER=4;
      end;

      snr(utt_ind,1) = ENV_NUMBER;
      snr(utt_ind,2) = 10*log10(sum(output.^2) ./ sum(sum(n.^2)));
      % Write WAV file
      output=output/max(abs(output));

      audiowrite([edir uname '.wav'],output,fs);
      disp([edir uname '.wav']);
    end
    csvwrite(['SNR_mvdr_all_data' mode '.csv'],snr);
  end
end
