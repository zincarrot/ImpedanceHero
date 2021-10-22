function plotreal_all(data)
f=data(1,:);
for k=2:size(data,1)
semilogx(f,real(data(k,:)));
hold on;
end
xlabel("f/Hz");
ylabel("Z'/\Omega");
end