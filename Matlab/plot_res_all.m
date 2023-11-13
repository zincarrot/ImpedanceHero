function plot_res_all(data,para)
c=jet(size(data,1));
for i=1:size(data,1)-1
    res1=data(i+1,:)-zcpe(para(i,:),data(1,:));
    res2=1./(1./res1-kl(para(i,:),data(1,:)));
    plotecplx(data(1,:),res2,1,c(i,:));
    hold on;
end

end