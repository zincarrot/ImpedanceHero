function mse = mse(Q,a,ch,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,fc,kl,n,r,rs,zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15)
%MSE
%    MSE = MSE(Q,A,CH,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,FC,KL,N,R,RS,ZDATA1,ZDATA2,ZDATA3,ZDATA4,ZDATA5,ZDATA6,ZDATA7,ZDATA8,ZDATA9,ZDATA10,ZDATA11,ZDATA12,ZDATA13,ZDATA14,ZDATA15)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    23-Feb-2021 16:35:54

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
t17 = -a;
t18 = 1.0./fc;
t19 = 1.0./r;
t20 = f1.*pi.*2.0i;
t21 = f2.*pi.*2.0i;
t22 = f3.*pi.*2.0i;
t23 = f4.*pi.*2.0i;
t24 = f5.*pi.*2.0i;
t25 = f6.*pi.*2.0i;
t26 = f7.*pi.*2.0i;
t27 = f8.*pi.*2.0i;
t28 = f9.*pi.*2.0i;
t29 = f10.*pi.*2.0i;
t30 = f11.*pi.*2.0i;
t31 = f12.*pi.*2.0i;
t32 = f13.*pi.*2.0i;
t33 = f14.*pi.*2.0i;
t34 = f15.*pi.*2.0i;
t35 = t17+1.0;
t36 = t20.^n;
t37 = t21.^n;
t38 = t22.^n;
t39 = t23.^n;
t40 = t24.^n;
t41 = t25.^n;
t42 = t26.^n;
t43 = t27.^n;
t44 = t28.^n;
t45 = t29.^n;
t46 = t30.^n;
t47 = t31.^n;
t48 = t32.^n;
t49 = t33.^n;
t50 = t34.^n;
t51 = f1.*t18.*1i;
t52 = f2.*t18.*1i;
t53 = f3.*t18.*1i;
t54 = f4.*t18.*1i;
t55 = f5.*t18.*1i;
t56 = f6.*t18.*1i;
t57 = f7.*t18.*1i;
t58 = f8.*t18.*1i;
t59 = f9.*t18.*1i;
t60 = f10.*t18.*1i;
t61 = f11.*t18.*1i;
t62 = f12.*t18.*1i;
t63 = f13.*t18.*1i;
t64 = f14.*t18.*1i;
t65 = f15.*t18.*1i;
t66 = Q.*t36;
t67 = Q.*t37;
t68 = Q.*t38;
t69 = Q.*t39;
t70 = Q.*t40;
t71 = Q.*t41;
t72 = Q.*t42;
t73 = Q.*t43;
t74 = Q.*t44;
t75 = Q.*t45;
t76 = Q.*t46;
t77 = Q.*t47;
t78 = Q.*t48;
t79 = Q.*t49;
t80 = Q.*t50;
t96 = t51.^t35;
t97 = t52.^t35;
t98 = t53.^t35;
t99 = t54.^t35;
t100 = t55.^t35;
t101 = t56.^t35;
t102 = t57.^t35;
t103 = t58.^t35;
t104 = t59.^t35;
t105 = t60.^t35;
t106 = t61.^t35;
t107 = t62.^t35;
t108 = t63.^t35;
t109 = t64.^t35;
t110 = t65.^t35;
t81 = t19+t66;
t82 = t19+t67;
t83 = t19+t68;
t84 = t19+t69;
t85 = t19+t70;
t86 = t19+t71;
t87 = t19+t72;
t88 = t19+t73;
t89 = t19+t74;
t90 = t19+t75;
t91 = t19+t76;
t92 = t19+t77;
t93 = t19+t78;
t94 = t19+t79;
t95 = t19+t80;
t126 = t96+1.0;
t127 = t97+1.0;
t128 = t98+1.0;
t129 = t99+1.0;
t130 = t100+1.0;
t131 = t101+1.0;
t132 = t102+1.0;
t133 = t103+1.0;
t134 = t104+1.0;
t135 = t105+1.0;
t136 = t106+1.0;
t137 = t107+1.0;
t138 = t108+1.0;
t139 = t109+1.0;
t140 = t110+1.0;
t111 = 1.0./t81;
t112 = 1.0./t82;
t113 = 1.0./t83;
t114 = 1.0./t84;
t115 = 1.0./t85;
t116 = 1.0./t86;
t117 = 1.0./t87;
t118 = 1.0./t88;
t119 = 1.0./t89;
t120 = 1.0./t90;
t121 = 1.0./t91;
t122 = 1.0./t92;
t123 = 1.0./t93;
t124 = 1.0./t94;
t125 = 1.0./t95;
t141 = 1.0./t126;
t142 = 1.0./t127;
t143 = 1.0./t128;
t144 = 1.0./t129;
t145 = 1.0./t130;
t146 = 1.0./t131;
t147 = 1.0./t132;
t148 = 1.0./t133;
t149 = 1.0./t134;
t150 = 1.0./t135;
t151 = 1.0./t136;
t152 = 1.0./t137;
t153 = 1.0./t138;
t154 = 1.0./t139;
t155 = 1.0./t140;
t156 = rs.*t141;
t157 = rs.*t142;
t158 = rs.*t143;
t159 = rs.*t144;
t160 = rs.*t145;
t161 = rs.*t146;
t162 = rs.*t147;
t163 = rs.*t148;
t164 = rs.*t149;
t165 = rs.*t150;
t166 = rs.*t151;
t167 = rs.*t152;
t168 = rs.*t153;
t169 = rs.*t154;
t170 = rs.*t155;
t171 = ch+t156;
t172 = ch+t157;
t173 = ch+t158;
t174 = ch+t159;
t175 = ch+t160;
t176 = ch+t161;
t177 = ch+t162;
t178 = ch+t163;
t179 = ch+t164;
t180 = ch+t165;
t181 = ch+t166;
t182 = ch+t167;
t183 = ch+t168;
t184 = ch+t169;
t185 = ch+t170;
t186 = t20.*t171;
t187 = t21.*t172;
t188 = t22.*t173;
t189 = t23.*t174;
t190 = t24.*t175;
t191 = t25.*t176;
t192 = t26.*t177;
t193 = t27.*t178;
t194 = t28.*t179;
t195 = t29.*t180;
t196 = t30.*t181;
t197 = t31.*t182;
t198 = t32.*t183;
t199 = t33.*t184;
t200 = t34.*t185;
t201 = kl+t186;
t202 = kl+t187;
t203 = kl+t188;
t204 = kl+t189;
t205 = kl+t190;
t206 = kl+t191;
t207 = kl+t192;
t208 = kl+t193;
t209 = kl+t194;
t210 = kl+t195;
t211 = kl+t196;
t212 = kl+t197;
t213 = kl+t198;
t214 = kl+t199;
t215 = kl+t200;
t216 = 1.0./t201;
t217 = 1.0./t202;
t218 = 1.0./t203;
t219 = 1.0./t204;
t220 = 1.0./t205;
t221 = 1.0./t206;
t222 = 1.0./t207;
t223 = 1.0./t208;
t224 = 1.0./t209;
t225 = 1.0./t210;
t226 = 1.0./t211;
t227 = 1.0./t212;
t228 = 1.0./t213;
t229 = 1.0./t214;
t230 = 1.0./t215;
t231 = t111+t216;
t232 = t112+t217;
t233 = t113+t218;
t234 = t114+t219;
t235 = t115+t220;
t236 = t116+t221;
t237 = t117+t222;
t238 = t118+t223;
t239 = t119+t224;
t240 = t120+t225;
t241 = t121+t226;
t242 = t122+t227;
t243 = t123+t228;
t244 = t124+t229;
t245 = t125+t230;
t246 = log(t231);
t247 = log(t232);
t248 = log(t233);
t249 = log(t234);
t250 = log(t235);
t251 = log(t236);
t252 = log(t237);
t253 = log(t238);
t254 = log(t239);
t255 = log(t240);
t256 = log(t241);
t257 = log(t242);
t258 = log(t243);
t259 = log(t244);
t260 = log(t245);
mse = ((conj(t2)-conj(t246)).*(t2-t246))./1.5e+1+((conj(t3)-conj(t247)).*(t3-t247))./1.5e+1+((conj(t4)-conj(t248)).*(t4-t248))./1.5e+1+((conj(t5)-conj(t249)).*(t5-t249))./1.5e+1+((conj(t6)-conj(t250)).*(t6-t250))./1.5e+1+((conj(t7)-conj(t251)).*(t7-t251))./1.5e+1+((conj(t8)-conj(t252)).*(t8-t252))./1.5e+1+((conj(t9)-conj(t253)).*(t9-t253))./1.5e+1+((conj(t10)-conj(t254)).*(t10-t254))./1.5e+1+((conj(t11)-conj(t255)).*(t11-t255))./1.5e+1+((conj(t12)-conj(t256)).*(t12-t256))./1.5e+1+((conj(t13)-conj(t257)).*(t13-t257))./1.5e+1+((conj(t14)-conj(t258)).*(t14-t258))./1.5e+1+((conj(t15)-conj(t259)).*(t15-t259))./1.5e+1+((conj(t16)-conj(t260)).*(t16-t260))./1.5e+1;