function processdata(date,s,batch)

f=importfreqdata(date+"_Parameter_Test_1.txt");
%%datasize=2;
%%data=zeros(datasize+1,15);
data(1,1:15)=f;
k=0;
while true
    try
    k=k+1;
    data(k+1,1:15)=importzdata(date+"_Sensor"+s+"_Test_"+k+".xlsx");
    disp(k);
    catch msg
        disp("data processing complete: "+k+" files in total");
        disp(msg);
        if k==1
            error("sensor "+s+" not found")
        end
        break;
    end
end
dataname="data_"+date+"_"+batch+"_"+s;
assignin('base',dataname,data);
end