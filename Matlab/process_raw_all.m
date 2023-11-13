function process_raw_all(comment)
files=dir("*_Parameter_Test_*.txt");
data_f=getfreq(files(1).name);
files=dir('*Test_*.xlsx');

sensor=0;
test=0;
for file=files'
    filename=file.name;
    parse=regexp(filename,'\d+','match');
    date=str2double(parse{1});
    sensor=max(sensor,str2double(parse{2}));
    test=max(test,str2double(parse{3}));
end
disp(sensor);
disp(test);
data=zeros(sensor+1,test+1,size(data_f,1));
for file=files'
    filename=file.name;
    disp(filename);
    parse=regexp(filename,'\d+','match');
    date=str2double(parse{1});
    sensor=str2double(parse{2});
    test=str2double(parse{3});
    data(sensor+1,1,:)=data_f;
    data_mean=getz(file.name);
    data(sensor+1,test+1,:)=data_mean;
end

for sensor=1:size(data,1)
    dataname="data_"+date+"_"+comment+"_"+(sensor-1);
    assignin('base',dataname,reshape(data(sensor,:,:),size(data,2),size(data,3)));
end
end