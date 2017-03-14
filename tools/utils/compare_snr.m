%clear;
function compare_snr()
addpath obj_evaluation;
utpath='../../../CHiME3/data/audio/16kHz/clean_data/'; % path to segmented utterances
enpath='../../../CHiME3/data/audio/16kHz/export_BLSTM/'; % path to segmented utterances
anpath='../../../CHiME3/data/annotations/'; % path to JSON annotations
resultpath='../../result/' ;

sets={'dt05'};
modes={'simu'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};

        % Read annotations
        mat=json2mat([anpath set '_' mode '.json']);
        real_mat=json2mat([anpath set '_real.json']);
        snr = zeros(length(mat), 3);
        for utt_ind=1:length(mat),
            utdir=[utpath]; % clean dir
			endir=[enpath set '_' lower(mat{utt_ind}.environment) '_' mode '/']; % enchance dir
      		  if ~exist(endir,'dir'),
                    system(['mkdir -p ' endir]);
            end
			if ~exist(resultpath,'dir'),
                system(['mkdir -p ' resultpath]);
            end
            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name];
            unaew=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            clean = ([utdir uname '.wav']);
            enhan = ([endir unaew '.wav']);
      			%transient estimation

            ENV_NUMBER=1;
            if strcmp(mat{utt_ind}.environment,'CAF'),
                ENV_NUMBER=2;
            elseif strcmp(mat{utt_ind}.environment,'PED'),
                ENV_NUMBER=3;
            elseif strcmp(mat{utt_ind}.environment,'STR'),
                ENV_NUMBER=4;
            end;
            peva = pesq(clean, enhan);
            [snr_mean, segsnr_mean] = comp_snr(clean, enhan);
            snr(utt_ind,1) = ENV_NUMBER;
            snr(utt_ind,2) = snr_mean;
            snr(utt_ind,3) = peva;

		    end
        csvwrite([resultpath 'SNR_export_BLSTM_' mode '.csv'],snr);
    end
end
