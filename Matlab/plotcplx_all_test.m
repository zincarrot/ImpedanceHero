function plotcplx_all_test(data)
for k=2:size(data,1)
    y=data(k,:);
plot(real(y),imag(y));
axis equal;
end
xlabel("Z'/\Omega");
ylabel("Z''/\Omega");
set(gca,'Ydir','reverse');
end