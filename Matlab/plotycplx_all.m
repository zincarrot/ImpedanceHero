function plotycplx_all(data)
c=jet(size(data,1));
for k=2:size(data,1)
    y=1./data(k,:);
plot(real(y),-imag(y),'Color',c(k,:));
hold on;
end
xlabel("Y'");
ylabel("-Y''");
end