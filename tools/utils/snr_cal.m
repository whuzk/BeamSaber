%clear;
function snr_cal(folder)
addpath obj_evaluation;
utpath='../../../CHiME3/data/audio/16kHz/clean_data/'; % path to segmented utterances
enpath=['../../../CHiME3/data/audio/16kHz/' folder '/']; % path to segmented utterances
anpath='../../../CHiME3/data/annotations/'; % path to JSON annotations
resultsnr='../../result/SNR/' ;
resultpesq='../../result/PESQ/' ;

sets={'dt05'};
modes={'simu'};
for set_ind=1:length(sets),
    set=sets{set_ind};
    for mode_ind=1:length(modes),
        mode=modes{mode_ind};

        % Read annotations
        mat=json2mat([anpath set '_' mode '.json']);
        snr = zeros(length(mat), 3);
		pesqValues = zeros(length(mat), 2);
        for utt_ind=1:length(mat),
            utdir=[utpath]; % clean dir
			endir=[enpath set '_' lower(mat{utt_ind}.environment) '_' mode '/']; % enchance dir

			if ~exist(resultsnr,'dir'),
                system(['mkdir -p ' resultsnr]);
            end
			if ~exist(resultpesq,'dir'),
                system(['mkdir -p ' resultpesq]);
            end

            uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name];
            unaew=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
            clean = ([utdir uname '.wav']);
            disp(clean);
            enhan = ([endir unaew '.wav']);
            disp(enhan);
      			%transient estimation

            ENV_NUMBER=1;
            if strcmp(mat{utt_ind}.environment,'CAF'),
                ENV_NUMBER=2;
            elseif strcmp(mat{utt_ind}.environment,'PED'),
                ENV_NUMBER=3;
            elseif strcmp(mat{utt_ind}.environment,'STR'),
                ENV_NUMBER=4;
            end;

            [snr_mean, segsnr_mean] = comp_snr(clean, enhan);
            snr(utt_ind,1) = ENV_NUMBER;
            snr(utt_ind,2) = snr_mean;
            snr(utt_ind,3) = segsnr_mean;

			[pesqVal] = pesq(clean, enhan);
			pesqValues(utt_ind,1) = ENV_NUMBER;
            pesqValues(utt_ind,2) = pesqVal;

		end

		csvwrite([resultsnr 'SNR_' folder '_' mode '.csv'],snr);
		csvwrite([resultpesq 'PESQ_' folder '_' mode '.csv'],pesqValues);
    end
end
