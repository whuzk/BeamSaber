function CHiME3_simulate_data(official)

if nargin < 1,
    official=true;
end
addpath ../utils;
upath='../../../CHiME3/data/audio/16kHz/isolated/'; % path to segmented utterances
%upath_ext = '../../data/audio/16kHz/isolated_ext/';
upath_ext = '../../../CHiME3/data/audio/16kHz/clean_dt/';
cpath='../../../CHiME3/data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../../../CHiME3/data/audio/16kHz/backgrounds/'; % path to noise backgrounds
apath=['../../../CHiME3/data/annotations/']; % path to JSON annotations
% new data path without server


% mic number 0,1,2,3,4,5
nchan=5;

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen_sub=256; % STFT window length in samples
blen_sub=4000; % average block length in samples for speech subtraction (250 ms)
ntap_sub=12; % filter length in frames for speech subtraction (88 ms)
wlen_add=1024; % STFT window length in samples for speaker localization
del=-3; % minimum delay (0 for a causal filter)


%% Create simulated development and test datasets from booth recordings %%
sets={'dt05'};
for set_ind=1:length(sets),
    set=sets{set_ind};

    % Read official annotations
    if official,
        mat=json2mat([apath set '_simu.json']);

    % Create new (non-official) annotations
    else
        disp('recreate');
        mat=json2mat([apath set '_real.json']);
        clean_mat=json2mat([apath set '_bth.json']);
        for utt_ind=1:length(mat),
            for clean_ind=1:length(clean_mat), % match noisy utterance with same clean utterance (may be from a different speaker)
                if strcmp(clean_mat{clean_ind}.wsj_name,mat{utt_ind}.wsj_name),
                    break;
                end
            end
            noise_mat=mat{utt_ind};
            mat{utt_ind}=clean_mat{clean_ind};
            mat{utt_ind}.environment=noise_mat.environment;
            mat{utt_ind}.noise_wavfile=noise_mat.wavfile;
            dur=mat{utt_ind}.end-mat{utt_ind}.start;
            noise_dur=noise_mat.end-noise_mat.start;
            pbeg=round((dur-noise_dur)/2*16000)/16000;
            pend=round((dur-noise_dur)*16000)/16000-pbeg;
            mat{utt_ind}.noise_start=noise_mat.start-pbeg;
            mat{utt_ind}.noise_end=noise_mat.end+pend;
            mat{utt_ind}=orderfields(mat{utt_ind});
        end
        mat2json(mat,[apath set '_simu_new.json']);
    end

    % Loop over utterances
    for utt_ind=1:length(mat),
        if official,
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu/'];
            udir_ext=[upath_ext 'dt05_' lower(mat{utt_ind}.environment) '_simu/'];
        else
            udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu_new/'];
        end
        if ~exist(udir,'dir'),
            system(['mkdir -p ' udir]);
        end
        if ~exist(udir_ext,'dir'),
            system(['mkdir -p ' udir_ext]);
        end
        oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
        nname=mat{utt_ind}.noise_wavfile;
        uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
        tbeg=round(mat{utt_ind}.noise_start*16000)+1;
        tend=round(mat{utt_ind}.noise_end*16000);

        % Load WAV files
        o=wavread([upath set '_bth/' oname '.CH0.wav']);
        [r,fs]=wavread([cpath nname '.CH0.wav'],[tbeg tend]);
        nsampl=length(r);
        x=zeros(nsampl,nchan);
        for c=1:nchan,
            x(:,c)=wavread([cpath nname '.CH' int2str(c) '.wav'],[tbeg tend]);
        end

        % Compute the STFT (short window)
        R=stft_multi(r.',wlen_sub);
        X=stft_multi(x.',wlen_sub);

        % Estimate 88 ms impulse responses on 250 ms time blocks
        A=estimate_ir(R,X,blen_sub,ntap_sub,del);

        % Filter and subtract close-mic speech
        Y=apply_ir(A,R,del);
        y=istft_multi(Y,nsampl).';
        level=sum(sum(y.^2));
        n=x-y;

        % Compute the STFT (long window)
        O=stft_multi(o.',wlen_add);
        X=stft_multi(x.',wlen_add);
        [nbin,nfram] = size(O);

        % Localize and track the speaker
        [~,TDOAx]=localize(X);

        % Interpolate the spatial position over the duration of clean speech
        TDOA=zeros(nchan,nfram);
        for c=1:nchan,
            TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
        end

        % Filter clean speech
        Ysimu=zeros(nbin,nfram,nchan);
        for f=1:nbin,
            for t=1:nfram,
                Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
                Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
            end
        end
        ysimu=istft_multi(Ysimu,nsampl).';

        % Normalize level and add
        ysimu=sqrt(level/sum(sum(ysimu.^2)))*ysimu;
        xsimu=ysimu+n;
	      yy = 0;
        % Write WAV file
        for c=1:nchan,
          wavwrite(xsimu(:,c),fs,[udir uname '.CH' int2str(c) '.wav']);
          %audiowrite([udir_ext uname '.CH' int2str(c) '.Noise.wav'],n(:, c),fs);
          %audiowrite([udir_ext uname '.CH' int2str(c) '.Clean.wav'],ysimu(:, c), fs);
          yy = yy + ysimu(:,c);
        end
	      audiowrite([udir_ext uname '.Clean.wav'],yy, fs);
    end
end

return
