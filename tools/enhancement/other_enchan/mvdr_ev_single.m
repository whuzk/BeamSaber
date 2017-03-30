clear all;

udir = '/media/hipo/lento/workspace/BeamSaber/tools/enhancement/gev/data/new_dataset/216m/';
uname = '2m_pub_new';
Nsource = 2;
Nchan = 8;
Lwindow = 256;
Nfft = Lwindow;
overlap = 0.75;
powThresh = -10;
cmin = 6400; % minimum context duration (400 ms)
cmax = 12800; % maximum context duration (800 ms)
EMITERNUM = 20;
GAMMA = 20;

xsize = size(audioread([udir uname '.CH1.wav']));
disp([udir uname]);
x = zeros(xsize(1),Nchan);
xc = zeros(xsize(1),Nchan);
for clp = 1:Nchan,
    [x(:,clp),fs] = audioread([udir uname '.CH' int2str(clp) '.wav']);
    xc(:,clp) = x(:,clp) / norm(x(:,clp)) * norm(x(:,1));
end

% Check microphone failure
CLwindow = 2048;
Coverlap = 0.5;
CNfft = CLwindow;
[~,~,~,~,CspeechFrame] =  STFT(xc, CLwindow, Coverlap, CNfft);
CspeechFrame = permute(CspeechFrame, [3,1,2]);
NcorFFT = 2 * (CNfft-1);
c1 = fft(repmat(permute(CspeechFrame, [1,4,2,3]),[1 Nchan 1 1]), NcorFFT, 3);
c2 = fft(repmat(permute(CspeechFrame, [4,1,2,3]),[Nchan 1 1 1]), NcorFFT, 3);
corr = real(ifft(c1 .* conj(c2),NcorFFT,3));
corr = squeeze(max(corr, [], 3));
corrjudge = sum(corr, 2) - sum(bsxfun(@times,corr,eye(Nchan)),2);
corrjudge = squeeze(corrjudge);
corrjudge_rate = bsxfun(@rdivide, corrjudge, median(corrjudge,1));
fail = corrjudge_rate < 0.6;
fail = (sum(fail,2) > 1);
fail = fail';
if all(fail)
    fail = (sum(corrjudge_rate,2) == max(sum(corrjudge_rate,2)));
    fail = ~fail;
    fail = fail';
end
if (any(fail))
    disp(['find one failure: ',set,' ',mode,' ','uttInd ',num2str(uttInd),' ',uname,' : ',num2str(fail)]);
    % fprintf(micFailFid, [set,' ',mode,' ','uttInd ',num2str(uttInd),' ',uname,' : ',num2str(fail),'\n']);
end

% choose reference mic
[~,refMic] = max(sum(corrjudge(~fail,:),2));

Nchan = sum(~fail);
x = x(:,~fail);

[ftbin,Nframe,Nbin,Lspeech] =  STFT(x, Lwindow, overlap, Nfft);
% for GSC fixed beamformer
targetY = squeeze(mean(ftbin,1));

% correlation of X
XX = bsxfun(@times, permute(ftbin,[1,4,2,3]), conj(permute(ftbin,[4,1,2,3])));

Xcor = mean(XX, 4);
% Ncor = mean(XX(:,:,:,[1:10,Nframe-10:Nframe]),4);
% Load context (up to 5 s immediately preceding the utterance)
% noise = read_context(nchan, c_ind, Path, mode, mat, real_mat, uttInd, cmax, cmin);
softmask = cGaussMask(ftbin,2,XX,EMITERNUM);
disp('softmask');
disp(size(softmask));
Ncor = bsxfun(@rdivide, mean(bsxfun(@times, XX, permute(softmask(:,:,2),[3,4,1,2])),4),permute(mean(softmask(:,:,2),2), [2,3,1]));
Gcor = Xcor - Ncor;
disp(size(Gcor));
% estimated noise
noise_ = NOISE(ftbin, Gcor, Ncor, refMic);

%for bssMvdrEg
yBssMvdrEgFt = MVDR_EV(ftbin, Gcor, Ncor);
disp('Y');
disp(size(yBssMvdrEgFt));
yBssMvdrEg = ISTFT(yBssMvdrEgFt,Lwindow,overlap);
yBssMvdrEg = yBssMvdrEg / max(abs(yBssMvdrEg));
disp(sum(yBssMvdrEg));

noiseWrite = ISTFT(noise_, Lwindow, overlap);
noiseWrite = noiseWrite / max(abs(noiseWrite));
wavwrite(noiseWrite, 16000, 16, 'mvdr_evNoise.wav');
wavwrite(yBssMvdrEg, 16000, 16, 'mvdr_ev2.wav');


% fclose(micFailFid);
