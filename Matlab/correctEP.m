function z=correctEP(z1,f)


% if nargin==2
%     a1=2*pi*real(z1(1))*f(1);
%     a2=2*pi*real(z1(2))*f(2);
%     b1=2*pi*imag(z1(1))*f(1);
%     b2=2*pi*imag(z1(2))*f(2);
%     a=b1*a2^2+b1*b2^2-b2*a1^2-b2*b1^2;
%     b=a2^2+b2^2-a1^2-b1^2;
%     c0=b2-b1;
% c1=(-b-sqrt(b^2-4*a*c0))/(2*a);
% z=z1-c(c1,f);
% end

if nargin==2
    z=z1-c(1/(z1(14)*2*pi*15000),f);
end

if nargin==1
    f=z1(1,:);
m=size(z1,1);
n=size(z1,2);
z=zeros(m,n);
z(1,:)=f;
for k=2:m
    z(k,:)=correctEP(z1(k,:),f);
end
end

