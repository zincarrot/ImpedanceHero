function process_raw_all_stdscreen(comment,threshold,screenall)

if nargin<2
    threshold=0.1;
end

if nargin<3
    screenall=true;
end

files=dir("*_Parameter_Test_*.txt");
data_f=getfreq(files(1).name);
files=dir('*Test_*.xlsx');

sensor=0;
test=0;
for file=files'
    filename=file.name;
    
    parse=regexp(filename,'\d+','match');
    if length(parse)==3
        date=str2double(parse{1});
        sensor=max(sensor,str2double(parse{2}));
        test=max(test,str2double(parse{3}));
    elseif length(parse)==2
        date=str2double(parse{1});
        sensor=0;
        test=max(test,str2double(parse{2}));
    end
    if contains(filename,'post')
        continue;
    end
end
disp(sensor);
disp(test);
data=zeros(sensor+1,test+1,size(data_f,1));
for file=files'
    filename=file.name;
    disp(filename);
    parse=regexp(filename,'\d+','match');
    
    if length(parse)==3
        date=str2double(parse{1});
        sensor=max(sensor,str2double(parse{2}));
        test=max(test,str2double(parse{3}));
    elseif length(parse)==2
        date=str2double(parse{1});
        sensor=0;
        test=max(test,str2double(parse{2}));
    end
    if contains(filename,'post')
        continue;
    end
    data(sensor+1,1,:)=data_f;
    [data_mean,data_std]=getz(file.name);
    if screenall
    if any(data_std./data_mean>threshold)
        data(sensor+1,test+1,:)=nan;
    else
        data(sensor+1,test+1,:)=data_mean;
    end
    else
        data_mean(data_std./data_mean>threshold)=nan;
        data(sensor+1,test+1,:)=data_mean;
    end
end

for sensor=1:size(data,1)
    dataname="data_"+date+"_"+comment+"_"+(sensor-1);
    assignin('base',dataname,reshape(data(sensor,:,:),size(data,2),size(data,3)));
end
end