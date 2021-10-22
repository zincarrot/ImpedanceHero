function fmean=getmean(data)
fmean=zeros(2,size(data,2));
fmean(1,:)=data(1,:);
for k=1:size(data,2)
    fmean(2,k)=mean(data(2:end,k));
end
end