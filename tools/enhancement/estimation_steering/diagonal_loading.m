function diagonal_loading()

track = '6ch';
addpath ../../utils;
addpath ../other_enchan;
upath=['../../../../CHiME3/data/audio/16kHz/isolated_' track '_track/']; % path to segmented utterances
epath=['../../../../CHiME3/data/audio/16kHz/estimation_steering_' track '_track/']; % path to enhanced utterances
cpath='../../../../CHiME3/data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../../../../CHiME3/data/audio/16kHz/backgrounds/'; % path to noise backgrounds
apath='../../../../CHiME3/data/annotations/'; % path to JSON annotations
resultpath='../../../result/';
% track = '6ch';
if strcmp(track,'6ch'),
    nchan=5;
elseif strcmp(track,'2ch'),
    nchan=2;
else
    error('This code is not suitable for single-channel data');
end

% Define hyper-parameters
Nsource = 2;
EMITERNUM = 20;
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen = 1024; % STFT window length
regul=1e-3; % MVDR regularization factor
cmin=6400; % minimum context duration (400 ms)
cmax=12800; % maximum context duration (800 ms)
Lwindow = 256;
overlap = 0.75;
Nfft = Lwindow;

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

            % Gcor
            [nbin,Nframe,Nbin,Lspeech] =  STFT(x, Lwindow, overlap, Nfft);
            % for GSC fixed beamformer
            targetY = squeeze(mean(nbin,1));
            XX = bsxfun(@times, permute(nbin,[1,4,2,3]), conj(permute(nbin,[4,1,2,3])));
            Xcor = mean(XX, 4);
            softmask = cGaussMask(nbin, Nsource, XX, EMITERNUM);
            Ncor = bsxfun(@rdivide, mean(bsxfun(@times, XX, permute(softmask(:,:,2),[3,4,1,2])),4),permute(mean(softmask(:,:,2),2), [2,3,1]));
            Gcor = Xcor - Ncor;
            % MVDR cgmm estimation
            mvdr_cg = MVDR_EV(nbin, Gcor, Ncor);
            output = istft_multi(mvdr_cg, nsampl).';
            output = output / max(abs(output));

            % Write WAV file
            output=output/max(abs(output));
            audiowrite([edir uname '.wav'],output,fs);
        end
    end
end

return
