function CHiME4_enhance_data_gsc(track)

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

addpath ../utils;
addpath ../utils/ArrayToolbox;
upath=['../../data/audio/16kHz/isolated_' track '_track/']; % path to segmented utterances
epath=['../../data/audio/16kHz/enhanced_' track '_track/']; % path to enhanced utterances
apath='../../data/annotations/'; % path to JSON annotations
if strcmp(track,'6ch'),
    nchan=5;
elseif strcmp(track,'2ch'),
    nchan=2;
else
    error('This code is not suitable for single-channel data');
end

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail

sets={'dt05'};
modes={'real' 'simu'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};
        
        % Read annotations
        % get JSON list
        mat=json2mat([apath set '_' mode '.json']);
   
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
                                      
            % GSC beamforming           
            c = 345.6;  %  Speed of sound for recording
			tWin = 40e-3;  % Window size for block processing
			nWin = round(tWin*fs);  % Audio window size in samples
			N = nsampl;
			if nWin/2 ~= fix(nWin/2)  % Ensure samples are even for overlap and add
				nWin = nWin+1;
			end
			nInc = round(nWin/2);  % Window increment %50 overlap
			M = nchan;  % Number of microphones

			% Define microphone positions in centimeters
			xmic=[-10 0 10 -10 0 10]; % left to right axis
			ymic=[9.5 9.5 9.5 -9.5 -9.5 -9.5]; % bottom to top axis
			zmic=[0 -2 0 0 0 0]; % back to front axis
			
			m = vertcat(xmic, ymic);
			m = vertcat(m, zmic)/100; % cm -> m
			
			% Define grid of possible speaker positions in centimeters
			xres=46;
			xpos=linspace(-45,45,xres);
			yres=46;
			ypos=linspace(-45,45,yres);
			zres=4;
			zpos=linspace(15,45,zres);
			s = [xpos(1) ypos(1) zpos(1)]';
			s = s/100; % cm -> m			
			
			hwin = hann(nWin+1);  %  Tappering window for overlap and add
			hwin = hwin(1:end-1);  % Make adjustment so even windows align
			
			% Traditional GJBF
			% Notice that several parameters must be saved and recycled between
			% iterations to ensure that the final conditions from one audio
			% window become the initial conditions for the next.
			mu = .1;  order = 20;  beta = .9;  % LMS filter parameters
			p = 0;  q = 0;  % don't need these now, set to zero
			phi = [];  psi = [];  % CCAF bounds not needed
			K = [];  % MC NLMS norm threshold not needed
			bmWForce = [];  mcWForce = [];  % not locking taps right now
			snrThresh = [];  snrRate = [];  % no SNR thresholding right now
			snrInit = [];
			xZ = zeros(nInc, M);  % Current window of audio data
			b = zeros(nInc, 1);  % embedded DSB output
			z = zeros(nInc, M-1);  % BM output
			bmWall = [];  % BM LMS taps (not needed here still but need [])
			mcWall = [];  % MC LMS taps (initialize to [])
			y = zeros(N, 1); % traditional GJBF output

            denomTemp = zeros(N, 1);
			for i=1:nInc:N-nWin  % iterate over 20ms windows     
				xPrev = xZ;  xZ = x(i:i+nWin-1,:); % load audio
				% Call beamforming function for this window
				[dum, bmWall, mcWall, snrAll, b, z] = ...
					gjbf(xZ, fs, s, m, c, p, q, mu, order, beta, ...
						 phi, psi, K, xPrev, b, z, bmWall(:,:,end), ...
						 mcWall(:,:,end), snrThresh, snrRate, snrInit, ...
						 bmWForce, mcWForce);
					 y(i:i+nWin-1) = y(i:i+nWin-1) + dum.*hwin;
                    denomTemp(i:i+nWin-1) = denomTemp(i:i+nWin-1) + (b-dum).*hwin;
			end
			display(length(denomTemp));
            % SNR calculation
            ENV_NUMBER=1;
            if strcmp(mat{utt_ind}.environment,'CAF'),
                ENV_NUMBER=2;
            elseif strcmp(mat{utt_ind}.environment,'PED'),
                ENV_NUMBER=3;
            elseif strcmp(mat{utt_ind}.environment,'STR'),
                ENV_NUMBER=4;
            end;
            snr(utt_ind,1) = ENV_NUMBER;
            snr(utt_ind,2) = 10*log10(sum(y.^2) ./ sum(denomTemp.^2));
            
            % Write WAV file
            y=y/(sqrt(2)*max(y));
            audiowrite([edir uname '.wav'],y,fs);
        end
        csvwrite(['SNR_GSC_' mode '.csv'],snr);
%         display(snr);
    end
end

return
