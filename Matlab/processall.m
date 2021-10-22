function processall(date,batch)
s=-1;
while true
    try
        s=s+1;
        processdata(date,s,batch);
    catch msg
        disp("data processing complete: "+s+" sensors in total");
        disp(msg);
        break;
    end
end