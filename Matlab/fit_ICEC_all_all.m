function fit_ICEC_all_all(data,x0)
k=1;    
    while true
        try
            a=data(k,:,:);
            disp(k);
        catch msg
            disp(msg);
            break;
        end
        k=k+1;
    end
end