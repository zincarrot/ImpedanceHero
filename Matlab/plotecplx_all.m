function plotecplx_all(data)
f=data(1,:);
%c=jet(size(data,1));
for k=2:size(data,1)
    t=1./(1i*2*pi*f.*(data(k,:)));
plot(real(t),-imag(t));%,'Color',c(k,:));
hold on;
end
xlabel("C'/F");
ylabel("-C''/F");
end