function plotcplxlog_all(data)
for k=2:size(data,1)
    y=data(k,:);
loglog(real(y),-imag(y));
hold on;
end
xlabel("Z'/\Omega");
ylabel("-Z''/\Omega");
%%set(gca,'Ydir','reverse');
end