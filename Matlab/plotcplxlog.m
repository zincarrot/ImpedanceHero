function plotcplxlog(z)
loglog(real(z),imag(z));
xlabel("Z'/\Omega");
ylabel("Z''/\Omega");
set(gca,'Ydir','reverse');
axis equal;
end