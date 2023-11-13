function plot_zcc_all(data,para)
c=jet(size(data,1));
for i=1:size(data,1)-1
    plotecplx(data(1,:),zcc(para(i,:),data(1,:)),1,c(i,:));
    hold on;
end

end