%clear;
function compare_snr()

track='6ch';
utpath=['../../data/audio/16kHz/' fbf '_' track '_track/']; % path to segmented utterances
enpath=['../../data/audio/16kHz/' fbf '_transomlsa_' track '_track/']; % path to enhanced utterances
anpath='../../data/annotations/'; % path to JSON annotations

sets={'dt05'};
modes={'real' 'simu'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};

        % Read annotations
        mat=json2mat([anpath set '_' mode '.json']);
        real_mat=json2mat([anpath set '_real.json']);

        for utt_ind=1:length(mat),
            utdir=[utpath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            endir=[enpath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
      		  if ~exist(endir,'dir'),
                    system(['mkdir -p ' endir]);
            end
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            display([utdir uname '.wav']);
      			%transient estimation
      			[yIn,Fs]=audioread([utdir uname '.wav']);
      			num_UC_frames=40;%number of uncausal frames
      			[~,tEst]=trans_estimating_omlsa_UC(yIn,num_UC_frames);
      			audiowrite([endir uname '_trans.wav'],tEst,Fs)
      			%speech enhancement
      			trans_reducing_omlsa([utdir uname '.wav'],[endir uname '.wav'],[endir uname '_trans.wav']);
		    end
    end
end
