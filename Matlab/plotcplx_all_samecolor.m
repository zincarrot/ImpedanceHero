function plotcplx_all_samecolor(data,color)
%c=jet(size(data,1));
for k=2:size(data,1)
    y=data(k,:);
plot(real(y),imag(y),'Color',color);
hold on;
end
xlabel("Z'/\Omega");
ylabel("Z''/\Omega");
set(gca,'Ydir','reverse');
end