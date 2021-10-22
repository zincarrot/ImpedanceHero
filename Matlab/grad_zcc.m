function grad_zcc = grad_zcc(a,ch,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,fc,rs,zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15)
%GRAD_ZCC
%    GRAD_ZCC = GRAD_ZCC(A,CH,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,FC,RS,ZDATA1,ZDATA2,ZDATA3,ZDATA4,ZDATA5,ZDATA6,ZDATA7,ZDATA8,ZDATA9,ZDATA10,ZDATA11,ZDATA12,ZDATA13,ZDATA14,ZDATA15)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    03-Mar-2021 23:31:16

t2 = conj(a);
t3 = conj(ch);
t4 = conj(fc);
t5 = conj(rs);
t6 = log(zdata1);
t7 = log(zdata2);
t8 = log(zdata3);
t9 = log(zdata4);
t10 = log(zdata5);
t11 = log(zdata6);
t12 = log(zdata7);
t13 = log(zdata8);
t14 = log(zdata9);
t15 = log(zdata10);
t16 = log(zdata11);
t17 = log(zdata12);
t18 = log(zdata13);
t19 = log(zdata14);
t20 = log(zdata15);
t36 = 1.0./pi;
t37 = -a;
t38 = a-1.0;
t39 = 1.0./f1;
t40 = 1.0./f2;
t41 = 1.0./f3;
t42 = 1.0./f4;
t43 = 1.0./f5;
t44 = 1.0./f6;
t45 = 1.0./f7;
t46 = 1.0./f8;
t47 = 1.0./f9;
t48 = 1.0./f10;
t49 = 1.0./f11;
t50 = 1.0./f12;
t51 = 1.0./f13;
t52 = 1.0./f14;
t53 = 1.0./f15;
t54 = 1.0./fc;
t21 = conj(t6);
t22 = conj(t7);
t23 = conj(t8);
t24 = conj(t9);
t25 = conj(t10);
t26 = conj(t11);
t27 = conj(t12);
t28 = conj(t13);
t29 = conj(t14);
t30 = conj(t15);
t31 = conj(t16);
t32 = conj(t17);
t33 = conj(t18);
t34 = conj(t19);
t35 = conj(t20);
t55 = t54.^2;
t56 = t2-1.0;
t57 = 1.0./t4.^2;
t58 = t37+1.0;
t59 = f1.*t54.*1i;
t60 = f2.*t54.*1i;
t61 = f3.*t54.*1i;
t62 = f4.*t54.*1i;
t63 = f5.*t54.*1i;
t64 = f6.*t54.*1i;
t65 = f7.*t54.*1i;
t66 = f8.*t54.*1i;
t67 = f9.*t54.*1i;
t68 = f10.*t54.*1i;
t69 = f11.*t54.*1i;
t70 = f12.*t54.*1i;
t71 = f13.*t54.*1i;
t72 = f14.*t54.*1i;
t73 = f15.*t54.*1i;
t74 = log(t59);
t75 = log(t60);
t76 = log(t61);
t77 = log(t62);
t78 = log(t63);
t79 = log(t64);
t80 = log(t65);
t81 = log(t66);
t82 = log(t67);
t83 = log(t68);
t84 = log(t69);
t85 = log(t70);
t86 = log(t71);
t87 = log(t72);
t88 = log(t73);
t89 = t59.^t38;
t90 = t60.^t38;
t91 = t61.^t38;
t92 = t62.^t38;
t93 = t63.^t38;
t94 = t64.^t38;
t95 = t65.^t38;
t96 = t66.^t38;
t97 = t67.^t38;
t98 = t68.^t38;
t99 = t69.^t38;
t100 = t70.^t38;
t101 = t71.^t38;
t102 = t72.^t38;
t103 = t73.^t38;
t119 = t59.^t58;
t120 = t60.^t58;
t121 = t61.^t58;
t122 = t62.^t58;
t123 = t63.^t58;
t124 = t64.^t58;
t125 = t65.^t58;
t126 = t66.^t58;
t127 = t67.^t58;
t128 = t68.^t58;
t129 = t69.^t58;
t130 = t70.^t58;
t131 = t71.^t58;
t132 = t72.^t58;
t133 = t73.^t58;
t104 = conj(t89);
t105 = conj(t90);
t106 = conj(t91);
t107 = conj(t92);
t108 = conj(t93);
t109 = conj(t94);
t110 = conj(t95);
t111 = conj(t96);
t112 = conj(t97);
t113 = conj(t98);
t114 = conj(t99);
t115 = conj(t100);
t116 = conj(t101);
t117 = conj(t102);
t118 = conj(t103);
t134 = t119+1.0;
t135 = t120+1.0;
t136 = t121+1.0;
t137 = t122+1.0;
t138 = t123+1.0;
t139 = t124+1.0;
t140 = t125+1.0;
t141 = t126+1.0;
t142 = t127+1.0;
t143 = t128+1.0;
t144 = t129+1.0;
t145 = t130+1.0;
t146 = t131+1.0;
t147 = t132+1.0;
t148 = t133+1.0;
t149 = 1.0./t104;
t150 = 1.0./t105;
t151 = 1.0./t106;
t152 = 1.0./t107;
t153 = 1.0./t108;
t154 = 1.0./t109;
t155 = 1.0./t110;
t156 = 1.0./t111;
t157 = 1.0./t112;
t158 = 1.0./t113;
t159 = 1.0./t114;
t160 = 1.0./t115;
t161 = 1.0./t116;
t162 = 1.0./t117;
t163 = 1.0./t118;
t179 = 1.0./t134;
t181 = 1.0./t135;
t183 = 1.0./t136;
t185 = 1.0./t137;
t187 = 1.0./t138;
t189 = 1.0./t139;
t191 = 1.0./t140;
t193 = 1.0./t141;
t195 = 1.0./t142;
t197 = 1.0./t143;
t199 = 1.0./t144;
t201 = 1.0./t145;
t203 = 1.0./t146;
t205 = 1.0./t147;
t207 = 1.0./t148;
t164 = t149+1.0;
t165 = t150+1.0;
t166 = t151+1.0;
t167 = t152+1.0;
t168 = t153+1.0;
t169 = t154+1.0;
t170 = t155+1.0;
t171 = t156+1.0;
t172 = t157+1.0;
t173 = t158+1.0;
t174 = t159+1.0;
t175 = t160+1.0;
t176 = t161+1.0;
t177 = t162+1.0;
t178 = t163+1.0;
t180 = t179.^2;
t182 = t181.^2;
t184 = t183.^2;
t186 = t185.^2;
t188 = t187.^2;
t190 = t189.^2;
t192 = t191.^2;
t194 = t193.^2;
t196 = t195.^2;
t198 = t197.^2;
t200 = t199.^2;
t202 = t201.^2;
t204 = t203.^2;
t206 = t205.^2;
t208 = t207.^2;
t239 = rs.*t179;
t240 = rs.*t181;
t241 = rs.*t183;
t242 = rs.*t185;
t243 = rs.*t187;
t244 = rs.*t189;
t245 = rs.*t191;
t246 = rs.*t193;
t247 = rs.*t195;
t248 = rs.*t197;
t249 = rs.*t199;
t250 = rs.*t201;
t251 = rs.*t203;
t252 = rs.*t205;
t253 = rs.*t207;
t209 = 1.0./t164;
t211 = 1.0./t165;
t213 = 1.0./t166;
t215 = 1.0./t167;
t217 = 1.0./t168;
t219 = 1.0./t169;
t221 = 1.0./t170;
t223 = 1.0./t171;
t225 = 1.0./t172;
t227 = 1.0./t173;
t229 = 1.0./t174;
t231 = 1.0./t175;
t233 = 1.0./t176;
t235 = 1.0./t177;
t237 = 1.0./t178;
t254 = ch+t239;
t255 = ch+t240;
t256 = ch+t241;
t257 = ch+t242;
t258 = ch+t243;
t259 = ch+t244;
t260 = ch+t245;
t261 = ch+t246;
t262 = ch+t247;
t263 = ch+t248;
t264 = ch+t249;
t265 = ch+t250;
t266 = ch+t251;
t267 = ch+t252;
t268 = ch+t253;
t210 = t209.^2;
t212 = t211.^2;
t214 = t213.^2;
t216 = t215.^2;
t218 = t217.^2;
t220 = t219.^2;
t222 = t221.^2;
t224 = t223.^2;
t226 = t225.^2;
t228 = t227.^2;
t230 = t229.^2;
t232 = t231.^2;
t234 = t233.^2;
t236 = t235.^2;
t238 = t237.^2;
t269 = t5.*t209;
t270 = t5.*t211;
t271 = t5.*t213;
t272 = t5.*t215;
t273 = t5.*t217;
t274 = t5.*t219;
t275 = t5.*t221;
t276 = t5.*t223;
t277 = t5.*t225;
t278 = t5.*t227;
t279 = t5.*t229;
t280 = t5.*t231;
t281 = t5.*t233;
t282 = t5.*t235;
t283 = t5.*t237;
t284 = 1.0./t254;
t285 = 1.0./t255;
t286 = 1.0./t256;
t287 = 1.0./t257;
t288 = 1.0./t258;
t289 = 1.0./t259;
t290 = 1.0./t260;
t291 = 1.0./t261;
t292 = 1.0./t262;
t293 = 1.0./t263;
t294 = 1.0./t264;
t295 = 1.0./t265;
t296 = 1.0./t266;
t297 = 1.0./t267;
t298 = 1.0./t268;
t299 = t3+t269;
t300 = t3+t270;
t301 = t3+t271;
t302 = t3+t272;
t303 = t3+t273;
t304 = t3+t274;
t305 = t3+t275;
t306 = t3+t276;
t307 = t3+t277;
t308 = t3+t278;
t309 = t3+t279;
t310 = t3+t280;
t311 = t3+t281;
t312 = t3+t282;
t313 = t3+t283;
t329 = t36.*t39.*t284.*5.0e-1i;
t330 = t36.*t40.*t285.*5.0e-1i;
t331 = t36.*t41.*t286.*5.0e-1i;
t332 = t36.*t42.*t287.*5.0e-1i;
t333 = t36.*t43.*t288.*5.0e-1i;
t334 = t36.*t44.*t289.*5.0e-1i;
t335 = t36.*t45.*t290.*5.0e-1i;
t336 = t36.*t46.*t291.*5.0e-1i;
t337 = t36.*t47.*t292.*5.0e-1i;
t338 = t36.*t48.*t293.*5.0e-1i;
t339 = t36.*t49.*t294.*5.0e-1i;
t340 = t36.*t50.*t295.*5.0e-1i;
t341 = t36.*t51.*t296.*5.0e-1i;
t342 = t36.*t52.*t297.*5.0e-1i;
t343 = t36.*t53.*t298.*5.0e-1i;
t314 = 1.0./t299;
t315 = 1.0./t300;
t316 = 1.0./t301;
t317 = 1.0./t302;
t318 = 1.0./t303;
t319 = 1.0./t304;
t320 = 1.0./t305;
t321 = 1.0./t306;
t322 = 1.0./t307;
t323 = 1.0./t308;
t324 = 1.0./t309;
t325 = 1.0./t310;
t326 = 1.0./t311;
t327 = 1.0./t312;
t328 = 1.0./t313;
t344 = -t329;
t345 = -t330;
t346 = -t331;
t347 = -t332;
t348 = -t333;
t349 = -t334;
t350 = -t335;
t351 = -t336;
t352 = -t337;
t353 = -t338;
t354 = -t339;
t355 = -t340;
t356 = -t341;
t357 = -t342;
t358 = -t343;
t359 = log(t344);
t360 = log(t345);
t361 = log(t346);
t362 = log(t347);
t363 = log(t348);
t364 = log(t349);
t365 = log(t350);
t366 = log(t351);
t367 = log(t352);
t368 = log(t353);
t369 = log(t354);
t370 = log(t355);
t371 = log(t356);
t372 = log(t357);
t373 = log(t358);
t374 = conj(t359);
t375 = conj(t360);
t376 = conj(t361);
t377 = conj(t362);
t378 = conj(t363);
t379 = conj(t364);
t380 = conj(t365);
t381 = conj(t366);
t382 = conj(t367);
t383 = conj(t368);
t384 = conj(t369);
t385 = conj(t370);
t386 = conj(t371);
t387 = conj(t372);
t388 = conj(t373);
t389 = -t359;
t390 = -t360;
t391 = -t361;
t392 = -t362;
t393 = -t363;
t394 = -t364;
t395 = -t365;
t396 = -t366;
t397 = -t367;
t398 = -t368;
t399 = -t369;
t400 = -t370;
t401 = -t371;
t402 = -t372;
t403 = -t373;
t404 = -t374;
t405 = -t375;
t406 = -t376;
t407 = -t377;
t408 = -t378;
t409 = -t379;
t410 = -t380;
t411 = -t381;
t412 = -t382;
t413 = -t383;
t414 = -t384;
t415 = -t385;
t416 = -t386;
t417 = -t387;
t418 = -t388;
t419 = t6+t389;
t420 = t7+t390;
t421 = t8+t391;
t422 = t9+t392;
t423 = t10+t393;
t424 = t11+t394;
t425 = t12+t395;
t426 = t13+t396;
t427 = t14+t397;
t428 = t15+t398;
t429 = t16+t399;
t430 = t17+t400;
t431 = t18+t401;
t432 = t19+t402;
t433 = t20+t403;
t434 = t21+t404;
t435 = t22+t405;
t436 = t23+t406;
t437 = t24+t407;
t438 = t25+t408;
t439 = t26+t409;
t440 = t27+t410;
t441 = t28+t411;
t442 = t29+t412;
t443 = t30+t413;
t444 = t31+t414;
t445 = t32+t415;
t446 = t33+t416;
t447 = t34+t417;
t448 = t35+t418;
grad_zcc = [real(rs.*t74.*t119.*t180.*t284.*t434)./1.5e+1+real(rs.*t75.*t120.*t182.*t285.*t435)./1.5e+1+real(rs.*t76.*t121.*t184.*t286.*t436)./1.5e+1+real(rs.*t77.*t122.*t186.*t287.*t437)./1.5e+1+real(rs.*t78.*t123.*t188.*t288.*t438)./1.5e+1+real(rs.*t79.*t124.*t190.*t289.*t439)./1.5e+1+real(rs.*t80.*t125.*t192.*t290.*t440)./1.5e+1+real(rs.*t81.*t126.*t194.*t291.*t441)./1.5e+1+real(rs.*t82.*t127.*t196.*t292.*t442)./1.5e+1+real(rs.*t83.*t128.*t198.*t293.*t443)./1.5e+1+real(rs.*t84.*t129.*t200.*t294.*t444)./1.5e+1+real(rs.*t85.*t130.*t202.*t295.*t445)./1.5e+1+real(rs.*t86.*t131.*t204.*t296.*t446)./1.5e+1+real(rs.*t87.*t132.*t206.*t297.*t447)./1.5e+1+real(rs.*t88.*t133.*t208.*t298.*t448)./1.5e+1+real(t5.*t149.*t210.*t314.*t419.*conj(t74))./1.5e+1+real(t5.*t150.*t212.*t315.*t420.*conj(t75))./1.5e+1+real(t5.*t151.*t214.*t316.*t421.*conj(t76))./1.5e+1+real(t5.*t152.*t216.*t317.*t422.*conj(t77))./1.5e+1+real(t5.*t153.*t218.*t318.*t423.*conj(t78))./1.5e+1+real(t5.*t154.*t220.*t319.*t424.*conj(t79))./1.5e+1+real(t5.*t155.*t222.*t320.*t425.*conj(t80))./1.5e+1+real(t5.*t156.*t224.*t321.*t426.*conj(t81))./1.5e+1+real(t5.*t157.*t226.*t322.*t427.*conj(t82))./1.5e+1+real(t5.*t158.*t228.*t323.*t428.*conj(t83))./1.5e+1+real(t5.*t159.*t230.*t324.*t429.*conj(t84))./1.5e+1+real(t5.*t160.*t232.*t325.*t430.*conj(t85))./1.5e+1+real(t5.*t161.*t234.*t326.*t431.*conj(t86))./1.5e+1+real(t5.*t162.*t236.*t327.*t432.*conj(t87))./1.5e+1+real(t5.*t163.*t238.*t328.*t433.*conj(t88))./1.5e+1,real(t284.*t434)./1.5e+1+real(t285.*t435)./1.5e+1+real(t286.*t436)./1.5e+1+real(t287.*t437)./1.5e+1+real(t288.*t438)./1.5e+1+real(t289.*t439)./1.5e+1+real(t290.*t440)./1.5e+1+real(t291.*t441)./1.5e+1+real(t314.*t419)./1.5e+1+real(t292.*t442)./1.5e+1+real(t315.*t420)./1.5e+1+real(t293.*t443)./1.5e+1+real(t316.*t421)./1.5e+1+real(t294.*t444)./1.5e+1+real(t317.*t422)./1.5e+1+real(t295.*t445)./1.5e+1+real(t318.*t423)./1.5e+1+real(t296.*t446)./1.5e+1+real(t319.*t424)./1.5e+1+real(t297.*t447)./1.5e+1+real(t320.*t425)./1.5e+1+real(t298.*t448)./1.5e+1+real(t321.*t426)./1.5e+1+real(t322.*t427)./1.5e+1+real(t323.*t428)./1.5e+1+real(t324.*t429)./1.5e+1+real(t325.*t430)./1.5e+1+real(t326.*t431)./1.5e+1+real(t327.*t432)./1.5e+1+real(t328.*t433)./1.5e+1,real(f1.*rs.*t38.*t55.*t59.^t37.*t180.*t284.*t434.*-6.666666666666667e-2i)+real(f2.*rs.*t38.*t55.*t60.^t37.*t182.*t285.*t435.*-6.666666666666667e-2i)+real(f3.*rs.*t38.*t55.*t61.^t37.*t184.*t286.*t436.*-6.666666666666667e-2i)+real(f4.*rs.*t38.*t55.*t62.^t37.*t186.*t287.*t437.*-6.666666666666667e-2i)+real(f5.*rs.*t38.*t55.*t63.^t37.*t188.*t288.*t438.*-6.666666666666667e-2i)+real(f6.*rs.*t38.*t55.*t64.^t37.*t190.*t289.*t439.*-6.666666666666667e-2i)+real(f7.*rs.*t38.*t55.*t65.^t37.*t192.*t290.*t440.*-6.666666666666667e-2i)+real(f8.*rs.*t38.*t55.*t66.^t37.*t194.*t291.*t441.*-6.666666666666667e-2i)+real(f9.*rs.*t38.*t55.*t67.^t37.*t196.*t292.*t442.*-6.666666666666667e-2i)+real(f10.*rs.*t38.*t55.*t68.^t37.*t198.*t293.*t443.*-6.666666666666667e-2i)+real(f11.*rs.*t38.*t55.*t69.^t37.*t200.*t294.*t444.*-6.666666666666667e-2i)+real(f12.*rs.*t38.*t55.*t70.^t37.*t202.*t295.*t445.*-6.666666666666667e-2i)+real(f13.*rs.*t38.*t55.*t71.^t37.*t204.*t296.*t446.*-6.666666666666667e-2i)+real(f14.*rs.*t38.*t55.*t72.^t37.*t206.*t297.*t447.*-6.666666666666667e-2i)+real(f15.*rs.*t38.*t55.*t73.^t37.*t208.*t298.*t448.*-6.666666666666667e-2i)+real((t5.*t56.*t57.*t210.*t314.*t419.*conj(f1).*6.666666666666667e-2i)./conj(t59.^a))+real((t5.*t56.*t57.*t212.*t315.*t420.*conj(f2).*6.666666666666667e-2i)./conj(t60.^a))+real((t5.*t56.*t57.*t214.*t316.*t421.*conj(f3).*6.666666666666667e-2i)./conj(t61.^a))+real((t5.*t56.*t57.*t216.*t317.*t422.*conj(f4).*6.666666666666667e-2i)./conj(t62.^a))+real((t5.*t56.*t57.*t218.*t318.*t423.*conj(f5).*6.666666666666667e-2i)./conj(t63.^a))+real((t5.*t56.*t57.*t220.*t319.*t424.*conj(f6).*6.666666666666667e-2i)./conj(t64.^a))+real((t5.*t56.*t57.*t222.*t320.*t425.*conj(f7).*6.666666666666667e-2i)./conj(t65.^a))+real((t5.*t56.*t57.*t224.*t321.*t426.*conj(f8).*6.666666666666667e-2i)./conj(t66.^a))+real((t5.*t56.*t57.*t226.*t322.*t427.*conj(f9).*6.666666666666667e-2i)./conj(t67.^a))+real((t5.*t56.*t57.*t228.*t323.*t428.*conj(f10).*6.666666666666667e-2i)./conj(t68.^a))+real((t5.*t56.*t57.*t230.*t324.*t429.*conj(f11).*6.666666666666667e-2i)./conj(t69.^a))+real((t5.*t56.*t57.*t232.*t325.*t430.*conj(f12).*6.666666666666667e-2i)./conj(t70.^a))+real((t5.*t56.*t57.*t234.*t326.*t431.*conj(f13).*6.666666666666667e-2i)./conj(t71.^a))+real((t5.*t56.*t57.*t236.*t327.*t432.*conj(f14).*6.666666666666667e-2i)./conj(t72.^a))+real((t5.*t56.*t57.*t238.*t328.*t433.*conj(f15).*6.666666666666667e-2i)./conj(t73.^a)),real(t179.*t284.*t434)./1.5e+1+real(t181.*t285.*t435)./1.5e+1+real(t183.*t286.*t436)./1.5e+1+real(t185.*t287.*t437)./1.5e+1+real(t187.*t288.*t438)./1.5e+1+real(t189.*t289.*t439)./1.5e+1+real(t191.*t290.*t440)./1.5e+1+real(t193.*t291.*t441)./1.5e+1+real(t195.*t292.*t442)./1.5e+1+real(t197.*t293.*t443)./1.5e+1+real(t199.*t294.*t444)./1.5e+1+real(t201.*t295.*t445)./1.5e+1+real(t209.*t314.*t419)./1.5e+1+real(t203.*t296.*t446)./1.5e+1+real(t211.*t315.*t420)./1.5e+1+real(t205.*t297.*t447)./1.5e+1+real(t213.*t316.*t421)./1.5e+1+real(t207.*t298.*t448)./1.5e+1+real(t215.*t317.*t422)./1.5e+1+real(t217.*t318.*t423)./1.5e+1+real(t219.*t319.*t424)./1.5e+1+real(t221.*t320.*t425)./1.5e+1+real(t223.*t321.*t426)./1.5e+1+real(t225.*t322.*t427)./1.5e+1+real(t227.*t323.*t428)./1.5e+1+real(t229.*t324.*t429)./1.5e+1+real(t231.*t325.*t430)./1.5e+1+real(t233.*t326.*t431)./1.5e+1+real(t235.*t327.*t432)./1.5e+1+real(t237.*t328.*t433)./1.5e+1];
