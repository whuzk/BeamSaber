clear;

%transient estimation
%[yIn,Fs]=audioread('sample/123.wav');
%num_UC_frames=40;%number of uncausal frames
%[~,tEst]=trans_estimating_omlsa_UC(yIn,num_UC_frames);
%audiowrite('sample/2mtransient_est.wav',tEst,Fs)
%speech enhancement
trans_reducing_omlsa('sample/2m_0703model_enhanced.wav', ...
					'/home/hipo/workspace/BeamSaber/result/gev+trans_reducing_omlsa/2m_0703model_post.wav', ...
					'sample/2m_0703model_noise.wav');
%see the sound files of the estimated transient and the enhanced speech in
%the folder "sample"
