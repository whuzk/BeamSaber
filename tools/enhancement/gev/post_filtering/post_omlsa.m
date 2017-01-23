%clear;
function postfilter_omlsa(fbf)

track='6ch';
utpath=['../../data/audio/16kHz/' fbf '_' track '_track/']; % path to segmented utterances
enpath=['../../data/audio/16kHz/' fbf '_omlsa_' track '_track/']; % path to enhanced utterances
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
            udir=[utpath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
            edir=[enpath set '_' lower(mat{utt_ind}.environment) '_' mode '/'];
			if ~exist(edir,'dir'),
                system(['mkdir -p ' edir]);
            end
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            
			%speech enhancement
			[yo,yout]=omlsa([udir uname '.wav'],[edir uname '.wav']);
		
		end
    end
end
end
