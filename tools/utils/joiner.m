function joiner(path, filetype)

d = dir([path '/*.' filetype]);
output_dir = '/media/hipo/lento/Dataset/LibriSpeech/';
C = [];
cont = 0;
for i = 1:length(d)
    disp([path d(i).name]);
    [yy, fs] = audioread([path d(i).name]);
    C = [yy; C];
[a, ~] = size(C);
if a > 22449280
a = 0;
%disp(size(C));
audiowrite([output_dir 'comb-' int2str(cont) '.wav'], C, fs);
cont = cont + 1;
C = [];
%disp(size(C));
end

end
audiowrite([output_dir 'comb.wav'], C, fs);
end
