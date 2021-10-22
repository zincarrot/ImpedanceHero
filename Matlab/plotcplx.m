
function plotcplx(z)
plot(real(z),imag(z));
xlabel("Z'/\Omega");
ylabel("Z''/\Omega");
set(gca,'Ydir','reverse');
axis equal;
hold on;
end