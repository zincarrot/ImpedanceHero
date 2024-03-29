function mse_trans = mse_trans(ce,cw,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,rl,rw,zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15)
%MSE_TRANS
%    MSE_TRANS = MSE_TRANS(CE,CW,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,RL,RW,ZDATA1,ZDATA2,ZDATA3,ZDATA4,ZDATA5,ZDATA6,ZDATA7,ZDATA8,ZDATA9,ZDATA10,ZDATA11,ZDATA12,ZDATA13,ZDATA14,ZDATA15)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    02-Mar-2021 22:47:29

t2 = log(zdata1);
t3 = log(zdata2);
t4 = log(zdata3);
t5 = log(zdata4);
t6 = log(zdata5);
t7 = log(zdata6);
t8 = log(zdata7);
t9 = log(zdata8);
t10 = log(zdata9);
t11 = log(zdata10);
t12 = log(zdata11);
t13 = log(zdata12);
t14 = log(zdata13);
t15 = log(zdata14);
t16 = log(zdata15);
t17 = 1.0./pi;
t18 = 1.0./ce;
t19 = 1.0./f1;
t20 = 1.0./f2;
t21 = 1.0./f3;
t22 = 1.0./f4;
t23 = 1.0./f5;
t24 = 1.0./f6;
t25 = 1.0./f7;
t26 = 1.0./f8;
t27 = 1.0./f9;
t28 = 1.0./f10;
t29 = 1.0./f11;
t30 = 1.0./f12;
t31 = 1.0./f13;
t32 = 1.0./f14;
t33 = 1.0./f15;
t34 = 1.0./rw;
t35 = cw.*f1.*pi.*2.0i;
t36 = cw.*f2.*pi.*2.0i;
t37 = cw.*f3.*pi.*2.0i;
t38 = cw.*f4.*pi.*2.0i;
t39 = cw.*f5.*pi.*2.0i;
t40 = cw.*f6.*pi.*2.0i;
t41 = cw.*f7.*pi.*2.0i;
t42 = cw.*f8.*pi.*2.0i;
t43 = cw.*f9.*pi.*2.0i;
t44 = cw.*f10.*pi.*2.0i;
t45 = cw.*f11.*pi.*2.0i;
t46 = cw.*f12.*pi.*2.0i;
t47 = cw.*f13.*pi.*2.0i;
t48 = cw.*f14.*pi.*2.0i;
t49 = cw.*f15.*pi.*2.0i;
t50 = t34+t35;
t51 = t34+t36;
t52 = t34+t37;
t53 = t34+t38;
t54 = t34+t39;
t55 = t34+t40;
t56 = t34+t41;
t57 = t34+t42;
t58 = t34+t43;
t59 = t34+t44;
t60 = t34+t45;
t61 = t34+t46;
t62 = t34+t47;
t63 = t34+t48;
t64 = t34+t49;
t80 = t17.*t18.*t19.*5.0e-1i;
t81 = t17.*t18.*t20.*5.0e-1i;
t82 = t17.*t18.*t21.*5.0e-1i;
t83 = t17.*t18.*t22.*5.0e-1i;
t84 = t17.*t18.*t23.*5.0e-1i;
t85 = t17.*t18.*t24.*5.0e-1i;
t86 = t17.*t18.*t25.*5.0e-1i;
t87 = t17.*t18.*t26.*5.0e-1i;
t88 = t17.*t18.*t27.*5.0e-1i;
t89 = t17.*t18.*t28.*5.0e-1i;
t90 = t17.*t18.*t29.*5.0e-1i;
t91 = t17.*t18.*t30.*5.0e-1i;
t92 = t17.*t18.*t31.*5.0e-1i;
t93 = t17.*t18.*t32.*5.0e-1i;
t94 = t17.*t18.*t33.*5.0e-1i;
t65 = 1.0./t50;
t66 = 1.0./t51;
t67 = 1.0./t52;
t68 = 1.0./t53;
t69 = 1.0./t54;
t70 = 1.0./t55;
t71 = 1.0./t56;
t72 = 1.0./t57;
t73 = 1.0./t58;
t74 = 1.0./t59;
t75 = 1.0./t60;
t76 = 1.0./t61;
t77 = 1.0./t62;
t78 = 1.0./t63;
t79 = 1.0./t64;
t95 = -t80;
t96 = -t81;
t97 = -t82;
t98 = -t83;
t99 = -t84;
t100 = -t85;
t101 = -t86;
t102 = -t87;
t103 = -t88;
t104 = -t89;
t105 = -t90;
t106 = -t91;
t107 = -t92;
t108 = -t93;
t109 = -t94;
t110 = rl+t65+t95;
t111 = rl+t66+t96;
t112 = rl+t67+t97;
t113 = rl+t68+t98;
t114 = rl+t69+t99;
t115 = rl+t70+t100;
t116 = rl+t71+t101;
t117 = rl+t72+t102;
t118 = rl+t73+t103;
t119 = rl+t74+t104;
t120 = rl+t75+t105;
t121 = rl+t76+t106;
t122 = rl+t77+t107;
t123 = rl+t78+t108;
t124 = rl+t79+t109;
t125 = log(t110);
t126 = log(t111);
t127 = log(t112);
t128 = log(t113);
t129 = log(t114);
t130 = log(t115);
t131 = log(t116);
t132 = log(t117);
t133 = log(t118);
t134 = log(t119);
t135 = log(t120);
t136 = log(t121);
t137 = log(t122);
t138 = log(t123);
t139 = log(t124);
mse_trans = ((conj(t2)-conj(t125)).*(t2-t125))./1.5e+1+((conj(t3)-conj(t126)).*(t3-t126))./1.5e+1+((conj(t4)-conj(t127)).*(t4-t127))./1.5e+1+((conj(t5)-conj(t128)).*(t5-t128))./1.5e+1+((conj(t6)-conj(t129)).*(t6-t129))./1.5e+1+((conj(t7)-conj(t130)).*(t7-t130))./1.5e+1+((conj(t8)-conj(t131)).*(t8-t131))./1.5e+1+((conj(t9)-conj(t132)).*(t9-t132))./1.5e+1+((conj(t10)-conj(t133)).*(t10-t133))./1.5e+1+((conj(t11)-conj(t134)).*(t11-t134))./1.5e+1+((conj(t12)-conj(t135)).*(t12-t135))./1.5e+1+((conj(t13)-conj(t136)).*(t13-t136))./1.5e+1+((conj(t14)-conj(t137)).*(t14-t137))./1.5e+1+((conj(t15)-conj(t138)).*(t15-t138))./1.5e+1+((conj(t16)-conj(t139)).*(t16-t139))./1.5e+1;
