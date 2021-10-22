function ztot2 = ztot2(a,aep,ch,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,fc,fcep,kl,rs,rsep)
%ZTOT2
%    ZTOT2 = ZTOT2(A,AEP,CH,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,FC,FCEP,KL,RS,RSEP)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    24-Feb-2021 15:01:30

t2 = 1.0./pi;
t3 = -a;
t4 = -aep;
t5 = 1.0./fc;
t6 = 1.0./fcep;
t7 = 1.0./rsep;
t8 = t3+1.0;
t9 = t4+1.0;
ztot2 = [1.0./(kl+f1.*pi.*(ch+rs./((f1.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f1.*t6.*1i).^t9+1.0).*5.0e-1i)./f1,1.0./(kl+f2.*pi.*(ch+rs./((f2.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f2.*t6.*1i).^t9+1.0).*5.0e-1i)./f2,1.0./(kl+f3.*pi.*(ch+rs./((f3.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f3.*t6.*1i).^t9+1.0).*5.0e-1i)./f3,1.0./(kl+f4.*pi.*(ch+rs./((f4.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f4.*t6.*1i).^t9+1.0).*5.0e-1i)./f4,1.0./(kl+f5.*pi.*(ch+rs./((f5.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f5.*t6.*1i).^t9+1.0).*5.0e-1i)./f5,1.0./(kl+f6.*pi.*(ch+rs./((f6.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f6.*t6.*1i).^t9+1.0).*5.0e-1i)./f6,1.0./(kl+f7.*pi.*(ch+rs./((f7.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f7.*t6.*1i).^t9+1.0).*5.0e-1i)./f7,1.0./(kl+f8.*pi.*(ch+rs./((f8.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f8.*t6.*1i).^t9+1.0).*5.0e-1i)./f8,1.0./(kl+f9.*pi.*(ch+rs./((f9.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f9.*t6.*1i).^t9+1.0).*5.0e-1i)./f9,1.0./(kl+f10.*pi.*(ch+rs./((f10.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f10.*t6.*1i).^t9+1.0).*5.0e-1i)./f10,1.0./(kl+f11.*pi.*(ch+rs./((f11.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f11.*t6.*1i).^t9+1.0).*5.0e-1i)./f11,1.0./(kl+f12.*pi.*(ch+rs./((f12.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f12.*t6.*1i).^t9+1.0).*5.0e-1i)./f12,1.0./(kl+f13.*pi.*(ch+rs./((f13.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f13.*t6.*1i).^t9+1.0).*5.0e-1i)./f13,1.0./(kl+f14.*pi.*(ch+rs./((f14.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f14.*t6.*1i).^t9+1.0).*5.0e-1i)./f14,1.0./(kl+f15.*pi.*(ch+rs./((f15.*t5.*1i).^t8+1.0)).*2.0i)-(t2.*t7.*((f15.*t6.*1i).^t9+1.0).*5.0e-1i)./f15];
