chimeRoot = '/home/hipo/workspace/CHiME3/';
enhRoot = '/home/hipo/workspace/CHiME3/data/audio/16kHz/';
workRoot = '/home/hipo/workspace/CHiME3/TEST/BSS_OPI/';
dataRoot = '/home/hipo/workspace/CHiME3/';
addpath ../../utils
addpath enhancement

Path.isolated = [dataRoot,'data/audio/16kHz/isolated/'];                % isolated data is at different locations same as backgrounds data
Path.enhanced = [chimeRoot,'data/audio/16kHz/enhanced/'];
Path.embedded = [dataRoot,'data/audio/16kHz/embedded/'];
Path.backgrounds = [dataRoot,'data/audio/16kHz/backgrounds/'];
Path.annotations = [chimeRoot,'data/annotations/']; % path to JSON annotations

% add path to enhancement result folder data
Path.enhBssMvdrEg = [enhRoot, 'enhBssMvdrEg/'];

Path.enhBssNoise = [enhRoot,'enhBssNoise/'];
Path.enhBssMvdrRtfNoise = [enhRoot, 'enhBssMvdrRtfNoise/'];
Path.enhBssMvdrEgNoise = [enhRoot, 'enhBssMvdrEgNoise/'];
Path.enhBssPmwfNoise = [enhRoot, 'enhBssPmwfNoise/'];

enhDirs = {'enhBssMvdrEg'};
% default set are et05, dt05, tr05. use only dt05 to test the beamformer
sets={'dt05'};
envirs = {'bus','caf','ped','str'};
modes={'real','simu'};
for hlp = 1:length(enhDirs)
    for slp = 1:length(sets)
        for elp = 1:length(envirs)
            for mlp = 1:length(modes)
                tDir = [enhRoot enhDirs{hlp} '/' sets{slp} '_' envirs{elp} '_' modes{mlp} '/'];
                if ~exist(tDir,'dir')
                    mkdir(tDir);
                end
            end
        end
    end
end
