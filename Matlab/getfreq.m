function freq=getfreq(filename)
s=fileread(filename);
lines=strsplit(s,'\n');
for line=lines(2:end)
    splitline=strsplit(cell2mat(line),':');
    if cell2mat(splitline(1)) == "Signal Frequency (Hz)"
        freq=cell2mat(textscan(cell2mat(splitline(2)),'%.6f'));
        break;
    end
end
end