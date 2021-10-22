function mse_zcc = mse_zcc(a,ch,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,fc,rs,zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15)
%MSE_ZCC
%    MSE_ZCC = MSE_ZCC(A,CH,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,FC,RS,ZDATA1,ZDATA2,ZDATA3,ZDATA4,ZDATA5,ZDATA6,ZDATA7,ZDATA8,ZDATA9,ZDATA10,ZDATA11,ZDATA12,ZDATA13,ZDATA14,ZDATA15)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    03-Mar-2021 23:26:02

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
t18 = -a;
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
t34 = 1.0./fc;
t35 = t18+1.0;
t36 = f1.*t34.*1i;
t37 = f2.*t34.*1i;
t38 = f3.*t34.*1i;
t39 = f4.*t34.*1i;
t40 = f5.*t34.*1i;
t41 = f6.*t34.*1i;
t42 = f7.*t34.*1i;
t43 = f8.*t34.*1i;
t44 = f9.*t34.*1i;
t45 = f10.*t34.*1i;
t46 = f11.*t34.*1i;
t47 = f12.*t34.*1i;
t48 = f13.*t34.*1i;
t49 = f14.*t34.*1i;
t50 = f15.*t34.*1i;
t51 = t36.^t35;
t52 = t37.^t35;
t53 = t38.^t35;
t54 = t39.^t35;
t55 = t40.^t35;
t56 = t41.^t35;
t57 = t42.^t35;
t58 = t43.^t35;
t59 = t44.^t35;
t60 = t45.^t35;
t61 = t46.^t35;
t62 = t47.^t35;
t63 = t48.^t35;
t64 = t49.^t35;
t65 = t50.^t35;
t66 = t51+1.0;
t67 = t52+1.0;
t68 = t53+1.0;
t69 = t54+1.0;
t70 = t55+1.0;
t71 = t56+1.0;
t72 = t57+1.0;
t73 = t58+1.0;
t74 = t59+1.0;
t75 = t60+1.0;
t76 = t61+1.0;
t77 = t62+1.0;
t78 = t63+1.0;
t79 = t64+1.0;
t80 = t65+1.0;
t81 = 1.0./t66;
t82 = 1.0./t67;
t83 = 1.0./t68;
t84 = 1.0./t69;
t85 = 1.0./t70;
t86 = 1.0./t71;
t87 = 1.0./t72;
t88 = 1.0./t73;
t89 = 1.0./t74;
t90 = 1.0./t75;
t91 = 1.0./t76;
t92 = 1.0./t77;
t93 = 1.0./t78;
t94 = 1.0./t79;
t95 = 1.0./t80;
t96 = rs.*t81;
t97 = rs.*t82;
t98 = rs.*t83;
t99 = rs.*t84;
t100 = rs.*t85;
t101 = rs.*t86;
t102 = rs.*t87;
t103 = rs.*t88;
t104 = rs.*t89;
t105 = rs.*t90;
t106 = rs.*t91;
t107 = rs.*t92;
t108 = rs.*t93;
t109 = rs.*t94;
t110 = rs.*t95;
t111 = ch+t96;
t112 = ch+t97;
t113 = ch+t98;
t114 = ch+t99;
t115 = ch+t100;
t116 = ch+t101;
t117 = ch+t102;
t118 = ch+t103;
t119 = ch+t104;
t120 = ch+t105;
t121 = ch+t106;
t122 = ch+t107;
t123 = ch+t108;
t124 = ch+t109;
t125 = ch+t110;
t126 = 1.0./t111;
t127 = 1.0./t112;
t128 = 1.0./t113;
t129 = 1.0./t114;
t130 = 1.0./t115;
t131 = 1.0./t116;
t132 = 1.0./t117;
t133 = 1.0./t118;
t134 = 1.0./t119;
t135 = 1.0./t120;
t136 = 1.0./t121;
t137 = 1.0./t122;
t138 = 1.0./t123;
t139 = 1.0./t124;
t140 = 1.0./t125;
t141 = t17.*t19.*t126.*5.0e-1i;
t142 = t17.*t20.*t127.*5.0e-1i;
t143 = t17.*t21.*t128.*5.0e-1i;
t144 = t17.*t22.*t129.*5.0e-1i;
t145 = t17.*t23.*t130.*5.0e-1i;
t146 = t17.*t24.*t131.*5.0e-1i;
t147 = t17.*t25.*t132.*5.0e-1i;
t148 = t17.*t26.*t133.*5.0e-1i;
t149 = t17.*t27.*t134.*5.0e-1i;
t150 = t17.*t28.*t135.*5.0e-1i;
t151 = t17.*t29.*t136.*5.0e-1i;
t152 = t17.*t30.*t137.*5.0e-1i;
t153 = t17.*t31.*t138.*5.0e-1i;
t154 = t17.*t32.*t139.*5.0e-1i;
t155 = t17.*t33.*t140.*5.0e-1i;
t156 = -t141;
t157 = -t142;
t158 = -t143;
t159 = -t144;
t160 = -t145;
t161 = -t146;
t162 = -t147;
t163 = -t148;
t164 = -t149;
t165 = -t150;
t166 = -t151;
t167 = -t152;
t168 = -t153;
t169 = -t154;
t170 = -t155;
t171 = log(t156);
t172 = log(t157);
t173 = log(t158);
t174 = log(t159);
t175 = log(t160);
t176 = log(t161);
t177 = log(t162);
t178 = log(t163);
t179 = log(t164);
t180 = log(t165);
t181 = log(t166);
t182 = log(t167);
t183 = log(t168);
t184 = log(t169);
t185 = log(t170);
mse_zcc = ((conj(t2)-conj(t171)).*(t2-t171))./1.5e+1+((conj(t3)-conj(t172)).*(t3-t172))./1.5e+1+((conj(t4)-conj(t173)).*(t4-t173))./1.5e+1+((conj(t5)-conj(t174)).*(t5-t174))./1.5e+1+((conj(t6)-conj(t175)).*(t6-t175))./1.5e+1+((conj(t7)-conj(t176)).*(t7-t176))./1.5e+1+((conj(t8)-conj(t177)).*(t8-t177))./1.5e+1+((conj(t9)-conj(t178)).*(t9-t178))./1.5e+1+((conj(t10)-conj(t179)).*(t10-t179))./1.5e+1+((conj(t11)-conj(t180)).*(t11-t180))./1.5e+1+((conj(t12)-conj(t181)).*(t12-t181))./1.5e+1+((conj(t13)-conj(t182)).*(t13-t182))./1.5e+1+((conj(t14)-conj(t183)).*(t14-t183))./1.5e+1+((conj(t15)-conj(t184)).*(t15-t184))./1.5e+1+((conj(t16)-conj(t185)).*(t16-t185))./1.5e+1;
