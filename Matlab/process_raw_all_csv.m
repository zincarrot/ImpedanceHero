function process_raw_all_csv()
mkdir processed;
files=dir("*_Parameter_Test_*.txt");
data_f=getfreq(files(1).name);
files=dir('*Test_*.xlsx');
for file=files'
    disp(file)
    data_mean=getz(file.name);
    data(:,1)=data_f;
    data(:,2)=real(data_mean);
    data(:,3)=imag(data_mean);
    saveto=file.folder+"//processed//"+file.name+".csv";
    writematrix(data,saveto);
end
