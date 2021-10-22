function plotyimag_all(data)
f=data(1,:);
for k=2:size(data,1)
    y=1./data(k,:);
semilogx(f,-imag(y));
hold on;
end
xlabel("f/Hz");
ylabel("-Y''/\Omega^{-1}");
end