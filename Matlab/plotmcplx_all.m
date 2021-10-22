function plotmcplx_all(data)
f=data(1,:);
for k=2:size(data,1)
    t=1i*2*pi*f.*(data(k,:));
plot(real(t),-imag(t));
hold on;
end
xlabel("M'");
ylabel("-M''");
end