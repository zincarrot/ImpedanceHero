function data_sub=sub_sys(data,sys,mode)
if nargin<3
    data_sub=data-sys(2,:);
    data_sub(1,:)=data(1,:);
else
    if mode=='s'
        data_sub=data-sys(2,:);
        data_sub(1,:)=data(1,:);
    elseif mode=='p'
        data_sub=data;
        data_sub(2:end,:)=1./(1./data(2:end,:)-1./sys(2,:));
    elseif mode=='ss'
        data_sub=data;
        data_sub(2:end,:)=data(2:end,:)-sys(2:end,:);
    elseif mode=='pp'
        data_sub=data;
        data_sub(2:end,:)=1./(1./data(2:end,:)-sys(2:end,:));
    else 
        disp("mode not recognized!");
    end
end

end