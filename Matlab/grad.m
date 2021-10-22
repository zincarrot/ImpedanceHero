function grad = grad(Q,a,ch,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,fc,kl,n,r,rs,zdata1,zdata2,zdata3,zdata4,zdata5,zdata6,zdata7,zdata8,zdata9,zdata10,zdata11,zdata12,zdata13,zdata14,zdata15)
%GRAD
%    GRAD = GRAD(Q,A,CH,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,FC,KL,N,R,RS,ZDATA1,ZDATA2,ZDATA3,ZDATA4,ZDATA5,ZDATA6,ZDATA7,ZDATA8,ZDATA9,ZDATA10,ZDATA11,ZDATA12,ZDATA13,ZDATA14,ZDATA15)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    23-Feb-2021 18:08:23

t2 = conj(Q);
t3 = conj(a);
t4 = conj(ch);
t5 = conj(f1);
t6 = conj(f2);
t7 = conj(f3);
t8 = conj(f4);
t9 = conj(f5);
t10 = conj(f6);
t11 = conj(f7);
t12 = conj(f8);
t13 = conj(f9);
t14 = conj(f10);
t15 = conj(f11);
t16 = conj(f12);
t17 = conj(f13);
t18 = conj(f14);
t19 = conj(f15);
t20 = conj(fc);
t21 = conj(kl);
t22 = conj(r);
t23 = conj(rs);
t24 = log(zdata1);
t25 = log(zdata2);
t26 = log(zdata3);
t27 = log(zdata4);
t28 = log(zdata5);
t29 = log(zdata6);
t30 = log(zdata7);
t31 = log(zdata8);
t32 = log(zdata9);
t33 = log(zdata10);
t34 = log(zdata11);
t35 = log(zdata12);
t36 = log(zdata13);
t37 = log(zdata14);
t38 = log(zdata15);
t54 = -a;
t55 = a-1.0;
t56 = 1.0./fc;
t58 = 1.0./r;
t79 = f1.*pi.*2.0i;
t80 = f2.*pi.*2.0i;
t81 = f3.*pi.*2.0i;
t82 = f4.*pi.*2.0i;
t83 = f5.*pi.*2.0i;
t84 = f6.*pi.*2.0i;
t85 = f7.*pi.*2.0i;
t86 = f8.*pi.*2.0i;
t87 = f9.*pi.*2.0i;
t88 = f10.*pi.*2.0i;
t89 = f11.*pi.*2.0i;
t90 = f12.*pi.*2.0i;
t91 = f13.*pi.*2.0i;
t92 = f14.*pi.*2.0i;
t93 = f15.*pi.*2.0i;
t39 = conj(t24);
t40 = conj(t25);
t41 = conj(t26);
t42 = conj(t27);
t43 = conj(t28);
t44 = conj(t29);
t45 = conj(t30);
t46 = conj(t31);
t47 = conj(t32);
t48 = conj(t33);
t49 = conj(t34);
t50 = conj(t35);
t51 = conj(t36);
t52 = conj(t37);
t53 = conj(t38);
t57 = t56.^2;
t59 = t58.^2;
t60 = t3-1.0;
t61 = 1.0./t20.^2;
t62 = 1.0./t22;
t64 = -t24;
t65 = -t25;
t66 = -t26;
t67 = -t27;
t68 = -t28;
t69 = -t29;
t70 = -t30;
t71 = -t31;
t72 = -t32;
t73 = -t33;
t74 = -t34;
t75 = -t35;
t76 = -t36;
t77 = -t37;
t78 = -t38;
t94 = t54+1.0;
t95 = log(t79);
t96 = log(t80);
t97 = log(t81);
t98 = log(t82);
t99 = log(t83);
t100 = log(t84);
t101 = log(t85);
t102 = log(t86);
t103 = log(t87);
t104 = log(t88);
t105 = log(t89);
t106 = log(t90);
t107 = log(t91);
t108 = log(t92);
t109 = log(t93);
t110 = t79.^n;
t111 = t80.^n;
t112 = t81.^n;
t113 = t82.^n;
t114 = t83.^n;
t115 = t84.^n;
t116 = t85.^n;
t117 = t86.^n;
t118 = t87.^n;
t119 = t88.^n;
t120 = t89.^n;
t121 = t90.^n;
t122 = t91.^n;
t123 = t92.^n;
t124 = t93.^n;
t140 = f1.*t56.*1i;
t141 = f2.*t56.*1i;
t142 = f3.*t56.*1i;
t143 = f4.*t56.*1i;
t144 = f5.*t56.*1i;
t145 = f6.*t56.*1i;
t146 = f7.*t56.*1i;
t147 = f8.*t56.*1i;
t148 = f9.*t56.*1i;
t149 = f10.*t56.*1i;
t150 = f11.*t56.*1i;
t151 = f12.*t56.*1i;
t152 = f13.*t56.*1i;
t153 = f14.*t56.*1i;
t154 = f15.*t56.*1i;
t63 = t62.^2;
t125 = conj(t110);
t126 = conj(t111);
t127 = conj(t112);
t128 = conj(t113);
t129 = conj(t114);
t130 = conj(t115);
t131 = conj(t116);
t132 = conj(t117);
t133 = conj(t118);
t134 = conj(t119);
t135 = conj(t120);
t136 = conj(t121);
t137 = conj(t122);
t138 = conj(t123);
t139 = conj(t124);
t155 = Q.*t110;
t156 = Q.*t111;
t157 = Q.*t112;
t158 = Q.*t113;
t159 = Q.*t114;
t160 = Q.*t115;
t161 = Q.*t116;
t162 = Q.*t117;
t163 = Q.*t118;
t164 = Q.*t119;
t165 = Q.*t120;
t166 = Q.*t121;
t167 = Q.*t122;
t168 = Q.*t123;
t169 = Q.*t124;
t170 = log(t140);
t171 = log(t141);
t172 = log(t142);
t173 = log(t143);
t174 = log(t144);
t175 = log(t145);
t176 = log(t146);
t177 = log(t147);
t178 = log(t148);
t179 = log(t149);
t180 = log(t150);
t181 = log(t151);
t182 = log(t152);
t183 = log(t153);
t184 = log(t154);
t200 = t140.^t55;
t201 = t141.^t55;
t202 = t142.^t55;
t203 = t143.^t55;
t204 = t144.^t55;
t205 = t145.^t55;
t206 = t146.^t55;
t207 = t147.^t55;
t208 = t148.^t55;
t209 = t149.^t55;
t210 = t150.^t55;
t211 = t151.^t55;
t212 = t152.^t55;
t213 = t153.^t55;
t214 = t154.^t55;
t245 = t140.^t94;
t246 = t141.^t94;
t247 = t142.^t94;
t248 = t143.^t94;
t249 = t144.^t94;
t250 = t145.^t94;
t251 = t146.^t94;
t252 = t147.^t94;
t253 = t148.^t94;
t254 = t149.^t94;
t255 = t150.^t94;
t256 = t151.^t94;
t257 = t152.^t94;
t258 = t153.^t94;
t259 = t154.^t94;
t185 = t2.*t125;
t186 = t2.*t126;
t187 = t2.*t127;
t188 = t2.*t128;
t189 = t2.*t129;
t190 = t2.*t130;
t191 = t2.*t131;
t192 = t2.*t132;
t193 = t2.*t133;
t194 = t2.*t134;
t195 = t2.*t135;
t196 = t2.*t136;
t197 = t2.*t137;
t198 = t2.*t138;
t199 = t2.*t139;
t215 = conj(t200);
t216 = conj(t201);
t217 = conj(t202);
t218 = conj(t203);
t219 = conj(t204);
t220 = conj(t205);
t221 = conj(t206);
t222 = conj(t207);
t223 = conj(t208);
t224 = conj(t209);
t225 = conj(t210);
t226 = conj(t211);
t227 = conj(t212);
t228 = conj(t213);
t229 = conj(t214);
t230 = t58+t155;
t231 = t58+t156;
t232 = t58+t157;
t233 = t58+t158;
t234 = t58+t159;
t235 = t58+t160;
t236 = t58+t161;
t237 = t58+t162;
t238 = t58+t163;
t239 = t58+t164;
t240 = t58+t165;
t241 = t58+t166;
t242 = t58+t167;
t243 = t58+t168;
t244 = t58+t169;
t290 = t245+1.0;
t291 = t246+1.0;
t292 = t247+1.0;
t293 = t248+1.0;
t294 = t249+1.0;
t295 = t250+1.0;
t296 = t251+1.0;
t297 = t252+1.0;
t298 = t253+1.0;
t299 = t254+1.0;
t300 = t255+1.0;
t301 = t256+1.0;
t302 = t257+1.0;
t303 = t258+1.0;
t304 = t259+1.0;
t260 = 1.0./t230;
t262 = 1.0./t231;
t264 = 1.0./t232;
t266 = 1.0./t233;
t268 = 1.0./t234;
t270 = 1.0./t235;
t272 = 1.0./t236;
t274 = 1.0./t237;
t276 = 1.0./t238;
t278 = 1.0./t239;
t280 = 1.0./t240;
t282 = 1.0./t241;
t284 = 1.0./t242;
t286 = 1.0./t243;
t288 = 1.0./t244;
t305 = 1.0./t215;
t306 = 1.0./t216;
t307 = 1.0./t217;
t308 = 1.0./t218;
t309 = 1.0./t219;
t310 = 1.0./t220;
t311 = 1.0./t221;
t312 = 1.0./t222;
t313 = 1.0./t223;
t314 = 1.0./t224;
t315 = 1.0./t225;
t316 = 1.0./t226;
t317 = 1.0./t227;
t318 = 1.0./t228;
t319 = 1.0./t229;
t320 = t62+t185;
t321 = t62+t186;
t322 = t62+t187;
t323 = t62+t188;
t324 = t62+t189;
t325 = t62+t190;
t326 = t62+t191;
t327 = t62+t192;
t328 = t62+t193;
t329 = t62+t194;
t330 = t62+t195;
t331 = t62+t196;
t332 = t62+t197;
t333 = t62+t198;
t334 = t62+t199;
t380 = 1.0./t290;
t382 = 1.0./t291;
t384 = 1.0./t292;
t386 = 1.0./t293;
t388 = 1.0./t294;
t390 = 1.0./t295;
t392 = 1.0./t296;
t394 = 1.0./t297;
t396 = 1.0./t298;
t398 = 1.0./t299;
t400 = 1.0./t300;
t402 = 1.0./t301;
t404 = 1.0./t302;
t406 = 1.0./t303;
t408 = 1.0./t304;
t261 = t260.^2;
t263 = t262.^2;
t265 = t264.^2;
t267 = t266.^2;
t269 = t268.^2;
t271 = t270.^2;
t273 = t272.^2;
t275 = t274.^2;
t277 = t276.^2;
t279 = t278.^2;
t281 = t280.^2;
t283 = t282.^2;
t285 = t284.^2;
t287 = t286.^2;
t289 = t288.^2;
t335 = t305+1.0;
t336 = t306+1.0;
t337 = t307+1.0;
t338 = t308+1.0;
t339 = t309+1.0;
t340 = t310+1.0;
t341 = t311+1.0;
t342 = t312+1.0;
t343 = t313+1.0;
t344 = t314+1.0;
t345 = t315+1.0;
t346 = t316+1.0;
t347 = t317+1.0;
t348 = t318+1.0;
t349 = t319+1.0;
t350 = 1.0./t320;
t352 = 1.0./t321;
t354 = 1.0./t322;
t356 = 1.0./t323;
t358 = 1.0./t324;
t360 = 1.0./t325;
t362 = 1.0./t326;
t364 = 1.0./t327;
t366 = 1.0./t328;
t368 = 1.0./t329;
t370 = 1.0./t330;
t372 = 1.0./t331;
t374 = 1.0./t332;
t376 = 1.0./t333;
t378 = 1.0./t334;
t381 = t380.^2;
t383 = t382.^2;
t385 = t384.^2;
t387 = t386.^2;
t389 = t388.^2;
t391 = t390.^2;
t393 = t392.^2;
t395 = t394.^2;
t397 = t396.^2;
t399 = t398.^2;
t401 = t400.^2;
t403 = t402.^2;
t405 = t404.^2;
t407 = t406.^2;
t409 = t408.^2;
t440 = rs.*t380;
t441 = rs.*t382;
t442 = rs.*t384;
t443 = rs.*t386;
t444 = rs.*t388;
t445 = rs.*t390;
t446 = rs.*t392;
t447 = rs.*t394;
t448 = rs.*t396;
t449 = rs.*t398;
t450 = rs.*t400;
t451 = rs.*t402;
t452 = rs.*t404;
t453 = rs.*t406;
t454 = rs.*t408;
t351 = t350.^2;
t353 = t352.^2;
t355 = t354.^2;
t357 = t356.^2;
t359 = t358.^2;
t361 = t360.^2;
t363 = t362.^2;
t365 = t364.^2;
t367 = t366.^2;
t369 = t368.^2;
t371 = t370.^2;
t373 = t372.^2;
t375 = t374.^2;
t377 = t376.^2;
t379 = t378.^2;
t410 = 1.0./t335;
t412 = 1.0./t336;
t414 = 1.0./t337;
t416 = 1.0./t338;
t418 = 1.0./t339;
t420 = 1.0./t340;
t422 = 1.0./t341;
t424 = 1.0./t342;
t426 = 1.0./t343;
t428 = 1.0./t344;
t430 = 1.0./t345;
t432 = 1.0./t346;
t434 = 1.0./t347;
t436 = 1.0./t348;
t438 = 1.0./t349;
t455 = ch+t440;
t456 = ch+t441;
t457 = ch+t442;
t458 = ch+t443;
t459 = ch+t444;
t460 = ch+t445;
t461 = ch+t446;
t462 = ch+t447;
t463 = ch+t448;
t464 = ch+t449;
t465 = ch+t450;
t466 = ch+t451;
t467 = ch+t452;
t468 = ch+t453;
t469 = ch+t454;
t411 = t410.^2;
t413 = t412.^2;
t415 = t414.^2;
t417 = t416.^2;
t419 = t418.^2;
t421 = t420.^2;
t423 = t422.^2;
t425 = t424.^2;
t427 = t426.^2;
t429 = t428.^2;
t431 = t430.^2;
t433 = t432.^2;
t435 = t434.^2;
t437 = t436.^2;
t439 = t438.^2;
t470 = t23.*t410;
t471 = t23.*t412;
t472 = t23.*t414;
t473 = t23.*t416;
t474 = t23.*t418;
t475 = t23.*t420;
t476 = t23.*t422;
t477 = t23.*t424;
t478 = t23.*t426;
t479 = t23.*t428;
t480 = t23.*t430;
t481 = t23.*t432;
t482 = t23.*t434;
t483 = t23.*t436;
t484 = t23.*t438;
t500 = t79.*t455;
t501 = t80.*t456;
t502 = t81.*t457;
t503 = t82.*t458;
t504 = t83.*t459;
t505 = t84.*t460;
t506 = t85.*t461;
t507 = t86.*t462;
t508 = t87.*t463;
t509 = t88.*t464;
t510 = t89.*t465;
t511 = t90.*t466;
t512 = t91.*t467;
t513 = t92.*t468;
t514 = t93.*t469;
t485 = t4+t470;
t486 = t4+t471;
t487 = t4+t472;
t488 = t4+t473;
t489 = t4+t474;
t490 = t4+t475;
t491 = t4+t476;
t492 = t4+t477;
t493 = t4+t478;
t494 = t4+t479;
t495 = t4+t480;
t496 = t4+t481;
t497 = t4+t482;
t498 = t4+t483;
t499 = t4+t484;
t515 = kl+t500;
t516 = kl+t501;
t517 = kl+t502;
t518 = kl+t503;
t519 = kl+t504;
t520 = kl+t505;
t521 = kl+t506;
t522 = kl+t507;
t523 = kl+t508;
t524 = kl+t509;
t525 = kl+t510;
t526 = kl+t511;
t527 = kl+t512;
t528 = kl+t513;
t529 = kl+t514;
t530 = t5.*t485.*pi.*2.0i;
t531 = t6.*t486.*pi.*2.0i;
t532 = t7.*t487.*pi.*2.0i;
t533 = t8.*t488.*pi.*2.0i;
t534 = t9.*t489.*pi.*2.0i;
t535 = t10.*t490.*pi.*2.0i;
t536 = t11.*t491.*pi.*2.0i;
t537 = t12.*t492.*pi.*2.0i;
t538 = t13.*t493.*pi.*2.0i;
t539 = t14.*t494.*pi.*2.0i;
t540 = t15.*t495.*pi.*2.0i;
t541 = t16.*t496.*pi.*2.0i;
t542 = t17.*t497.*pi.*2.0i;
t543 = t18.*t498.*pi.*2.0i;
t544 = t19.*t499.*pi.*2.0i;
t545 = 1.0./t515;
t547 = 1.0./t516;
t549 = 1.0./t517;
t551 = 1.0./t518;
t553 = 1.0./t519;
t555 = 1.0./t520;
t557 = 1.0./t521;
t559 = 1.0./t522;
t561 = 1.0./t523;
t563 = 1.0./t524;
t565 = 1.0./t525;
t567 = 1.0./t526;
t569 = 1.0./t527;
t571 = 1.0./t528;
t573 = 1.0./t529;
t546 = t545.^2;
t548 = t547.^2;
t550 = t549.^2;
t552 = t551.^2;
t554 = t553.^2;
t556 = t555.^2;
t558 = t557.^2;
t560 = t559.^2;
t562 = t561.^2;
t564 = t563.^2;
t566 = t565.^2;
t568 = t567.^2;
t570 = t569.^2;
t572 = t571.^2;
t574 = t573.^2;
t575 = -t530;
t576 = -t531;
t577 = -t532;
t578 = -t533;
t579 = -t534;
t580 = -t535;
t581 = -t536;
t582 = -t537;
t583 = -t538;
t584 = -t539;
t585 = -t540;
t586 = -t541;
t587 = -t542;
t588 = -t543;
t589 = -t544;
t635 = t260+t545;
t636 = t262+t547;
t637 = t264+t549;
t638 = t266+t551;
t639 = t268+t553;
t640 = t270+t555;
t641 = t272+t557;
t642 = t274+t559;
t643 = t276+t561;
t644 = t278+t563;
t645 = t280+t565;
t646 = t282+t567;
t647 = t284+t569;
t648 = t286+t571;
t649 = t288+t573;
t590 = t21+t575;
t591 = t21+t576;
t592 = t21+t577;
t593 = t21+t578;
t594 = t21+t579;
t595 = t21+t580;
t596 = t21+t581;
t597 = t21+t582;
t598 = t21+t583;
t599 = t21+t584;
t600 = t21+t585;
t601 = t21+t586;
t602 = t21+t587;
t603 = t21+t588;
t604 = t21+t589;
t650 = log(t635);
t651 = log(t636);
t652 = log(t637);
t653 = log(t638);
t654 = log(t639);
t655 = log(t640);
t656 = log(t641);
t657 = log(t642);
t658 = log(t643);
t659 = log(t644);
t660 = log(t645);
t661 = log(t646);
t662 = log(t647);
t663 = log(t648);
t664 = log(t649);
t680 = 1.0./t635;
t681 = 1.0./t636;
t682 = 1.0./t637;
t683 = 1.0./t638;
t684 = 1.0./t639;
t685 = 1.0./t640;
t686 = 1.0./t641;
t687 = 1.0./t642;
t688 = 1.0./t643;
t689 = 1.0./t644;
t690 = 1.0./t645;
t691 = 1.0./t646;
t692 = 1.0./t647;
t693 = 1.0./t648;
t694 = 1.0./t649;
t605 = 1.0./t590;
t607 = 1.0./t591;
t609 = 1.0./t592;
t611 = 1.0./t593;
t613 = 1.0./t594;
t615 = 1.0./t595;
t617 = 1.0./t596;
t619 = 1.0./t597;
t621 = 1.0./t598;
t623 = 1.0./t599;
t625 = 1.0./t600;
t627 = 1.0./t601;
t629 = 1.0./t602;
t631 = 1.0./t603;
t633 = 1.0./t604;
t665 = conj(t650);
t666 = conj(t651);
t667 = conj(t652);
t668 = conj(t653);
t669 = conj(t654);
t670 = conj(t655);
t671 = conj(t656);
t672 = conj(t657);
t673 = conj(t658);
t674 = conj(t659);
t675 = conj(t660);
t676 = conj(t661);
t677 = conj(t662);
t678 = conj(t663);
t679 = conj(t664);
t710 = t64+t650;
t711 = t65+t651;
t712 = t66+t652;
t713 = t67+t653;
t714 = t68+t654;
t715 = t69+t655;
t716 = t70+t656;
t717 = t71+t657;
t718 = t72+t658;
t719 = t73+t659;
t720 = t74+t660;
t721 = t75+t661;
t722 = t76+t662;
t723 = t77+t663;
t724 = t78+t664;
t606 = t605.^2;
t608 = t607.^2;
t610 = t609.^2;
t612 = t611.^2;
t614 = t613.^2;
t616 = t615.^2;
t618 = t617.^2;
t620 = t619.^2;
t622 = t621.^2;
t624 = t623.^2;
t626 = t625.^2;
t628 = t627.^2;
t630 = t629.^2;
t632 = t631.^2;
t634 = t633.^2;
t695 = -t665;
t696 = -t666;
t697 = -t667;
t698 = -t668;
t699 = -t669;
t700 = -t670;
t701 = -t671;
t702 = -t672;
t703 = -t673;
t704 = -t674;
t705 = -t675;
t706 = -t676;
t707 = -t677;
t708 = -t678;
t709 = -t679;
t740 = t350+t605;
t741 = t352+t607;
t742 = t354+t609;
t743 = t356+t611;
t744 = t358+t613;
t745 = t360+t615;
t746 = t362+t617;
t747 = t364+t619;
t748 = t366+t621;
t749 = t368+t623;
t750 = t370+t625;
t751 = t372+t627;
t752 = t374+t629;
t753 = t376+t631;
t754 = t378+t633;
t725 = t39+t695;
t726 = t40+t696;
t727 = t41+t697;
t728 = t42+t698;
t729 = t43+t699;
t730 = t44+t700;
t731 = t45+t701;
t732 = t46+t702;
t733 = t47+t703;
t734 = t48+t704;
t735 = t49+t705;
t736 = t50+t706;
t737 = t51+t707;
t738 = t52+t708;
t739 = t53+t709;
t755 = 1.0./t740;
t756 = 1.0./t741;
t757 = 1.0./t742;
t758 = 1.0./t743;
t759 = 1.0./t744;
t760 = 1.0./t745;
t761 = 1.0./t746;
t762 = 1.0./t747;
t763 = 1.0./t748;
t764 = 1.0./t749;
t765 = 1.0./t750;
t766 = 1.0./t751;
t767 = 1.0./t752;
t768 = 1.0./t753;
t769 = 1.0./t754;
grad = [real(t110.*t261.*t680.*t725)./1.5e+1+real(t111.*t263.*t681.*t726)./1.5e+1+real(t112.*t265.*t682.*t727)./1.5e+1+real(t113.*t267.*t683.*t728)./1.5e+1+real(t114.*t269.*t684.*t729)./1.5e+1+real(t115.*t271.*t685.*t730)./1.5e+1+real(t116.*t273.*t686.*t731)./1.5e+1+real(t117.*t275.*t687.*t732)./1.5e+1+real(t118.*t277.*t688.*t733)./1.5e+1+real(t119.*t279.*t689.*t734)./1.5e+1+real(t120.*t281.*t690.*t735)./1.5e+1+real(t121.*t283.*t691.*t736)./1.5e+1+real(t122.*t285.*t692.*t737)./1.5e+1+real(t123.*t287.*t693.*t738)./1.5e+1+real(t124.*t289.*t694.*t739)./1.5e+1-real(-t125.*t351.*t755.*(t24-t650))./1.5e+1-real(-t126.*t353.*t756.*(t25-t651))./1.5e+1-real(-t127.*t355.*t757.*(t26-t652))./1.5e+1-real(-t128.*t357.*t758.*(t27-t653))./1.5e+1-real(-t129.*t359.*t759.*(t28-t654))./1.5e+1-real(-t130.*t361.*t760.*(t29-t655))./1.5e+1-real(-t131.*t363.*t761.*(t30-t656))./1.5e+1-real(-t132.*t365.*t762.*(t31-t657))./1.5e+1-real(-t133.*t367.*t763.*(t32-t658))./1.5e+1-real(-t134.*t369.*t764.*(t33-t659))./1.5e+1-real(-t135.*t371.*t765.*(t34-t660))./1.5e+1-real(-t136.*t373.*t766.*(t35-t661))./1.5e+1-real(-t137.*t375.*t767.*(t36-t662))./1.5e+1-real(-t138.*t377.*t768.*(t37-t663))./1.5e+1-real(-t139.*t379.*t769.*(t38-t664))./1.5e+1,real(t5.*t23.*t305.*t411.*t606.*t755.*pi.*conj(t170).*(t24-t650).*-1.333333333333333e-1i)+real(t6.*t23.*t306.*t413.*t608.*t756.*pi.*conj(t171).*(t25-t651).*-1.333333333333333e-1i)+real(t7.*t23.*t307.*t415.*t610.*t757.*pi.*conj(t172).*(t26-t652).*-1.333333333333333e-1i)+real(t8.*t23.*t308.*t417.*t612.*t758.*pi.*conj(t173).*(t27-t653).*-1.333333333333333e-1i)+real(t9.*t23.*t309.*t419.*t614.*t759.*pi.*conj(t174).*(t28-t654).*-1.333333333333333e-1i)+real(t10.*t23.*t310.*t421.*t616.*t760.*pi.*conj(t175).*(t29-t655).*-1.333333333333333e-1i)+real(t11.*t23.*t311.*t423.*t618.*t761.*pi.*conj(t176).*(t30-t656).*-1.333333333333333e-1i)+real(t12.*t23.*t312.*t425.*t620.*t762.*pi.*conj(t177).*(t31-t657).*-1.333333333333333e-1i)+real(t13.*t23.*t313.*t427.*t622.*t763.*pi.*conj(t178).*(t32-t658).*-1.333333333333333e-1i)+real(t14.*t23.*t314.*t429.*t624.*t764.*pi.*conj(t179).*(t33-t659).*-1.333333333333333e-1i)+real(t15.*t23.*t315.*t431.*t626.*t765.*pi.*conj(t180).*(t34-t660).*-1.333333333333333e-1i)+real(t16.*t23.*t316.*t433.*t628.*t766.*pi.*conj(t181).*(t35-t661).*-1.333333333333333e-1i)+real(t17.*t23.*t317.*t435.*t630.*t767.*pi.*conj(t182).*(t36-t662).*-1.333333333333333e-1i)+real(t18.*t23.*t318.*t437.*t632.*t768.*pi.*conj(t183).*(t37-t663).*-1.333333333333333e-1i)+real(t19.*t23.*t319.*t439.*t634.*t769.*pi.*conj(t184).*(t38-t664).*-1.333333333333333e-1i)+real(f1.*rs.*t170.*t245.*t381.*t546.*t680.*t725.*pi.*1.333333333333333e-1i)+real(f2.*rs.*t171.*t246.*t383.*t548.*t681.*t726.*pi.*1.333333333333333e-1i)+real(f3.*rs.*t172.*t247.*t385.*t550.*t682.*t727.*pi.*1.333333333333333e-1i)+real(f4.*rs.*t173.*t248.*t387.*t552.*t683.*t728.*pi.*1.333333333333333e-1i)+real(f5.*rs.*t174.*t249.*t389.*t554.*t684.*t729.*pi.*1.333333333333333e-1i)+real(f6.*rs.*t175.*t250.*t391.*t556.*t685.*t730.*pi.*1.333333333333333e-1i)+real(f7.*rs.*t176.*t251.*t393.*t558.*t686.*t731.*pi.*1.333333333333333e-1i)+real(f8.*rs.*t177.*t252.*t395.*t560.*t687.*t732.*pi.*1.333333333333333e-1i)+real(f9.*rs.*t178.*t253.*t397.*t562.*t688.*t733.*pi.*1.333333333333333e-1i)+real(f10.*rs.*t179.*t254.*t399.*t564.*t689.*t734.*pi.*1.333333333333333e-1i)+real(f11.*rs.*t180.*t255.*t401.*t566.*t690.*t735.*pi.*1.333333333333333e-1i)+real(f12.*rs.*t181.*t256.*t403.*t568.*t691.*t736.*pi.*1.333333333333333e-1i)+real(f13.*rs.*t182.*t257.*t405.*t570.*t692.*t737.*pi.*1.333333333333333e-1i)+real(f14.*rs.*t183.*t258.*t407.*t572.*t693.*t738.*pi.*1.333333333333333e-1i)+real(f15.*rs.*t184.*t259.*t409.*t574.*t694.*t739.*pi.*1.333333333333333e-1i),real(t5.*t606.*t755.*pi.*(t24-t650).*-1.333333333333333e-1i)+real(t6.*t608.*t756.*pi.*(t25-t651).*-1.333333333333333e-1i)+real(t7.*t610.*t757.*pi.*(t26-t652).*-1.333333333333333e-1i)+real(t8.*t612.*t758.*pi.*(t27-t653).*-1.333333333333333e-1i)+real(t9.*t614.*t759.*pi.*(t28-t654).*-1.333333333333333e-1i)+real(t10.*t616.*t760.*pi.*(t29-t655).*-1.333333333333333e-1i)+real(t11.*t618.*t761.*pi.*(t30-t656).*-1.333333333333333e-1i)+real(t12.*t620.*t762.*pi.*(t31-t657).*-1.333333333333333e-1i)+real(t13.*t622.*t763.*pi.*(t32-t658).*-1.333333333333333e-1i)+real(t14.*t624.*t764.*pi.*(t33-t659).*-1.333333333333333e-1i)+real(t15.*t626.*t765.*pi.*(t34-t660).*-1.333333333333333e-1i)+real(t16.*t628.*t766.*pi.*(t35-t661).*-1.333333333333333e-1i)+real(t17.*t630.*t767.*pi.*(t36-t662).*-1.333333333333333e-1i)+real(t18.*t632.*t768.*pi.*(t37-t663).*-1.333333333333333e-1i)+real(t19.*t634.*t769.*pi.*(t38-t664).*-1.333333333333333e-1i)+real(f1.*t546.*t680.*t725.*pi.*1.333333333333333e-1i)+real(f2.*t548.*t681.*t726.*pi.*1.333333333333333e-1i)+real(f3.*t550.*t682.*t727.*pi.*1.333333333333333e-1i)+real(f4.*t552.*t683.*t728.*pi.*1.333333333333333e-1i)+real(f5.*t554.*t684.*t729.*pi.*1.333333333333333e-1i)+real(f6.*t556.*t685.*t730.*pi.*1.333333333333333e-1i)+real(f7.*t558.*t686.*t731.*pi.*1.333333333333333e-1i)+real(f8.*t560.*t687.*t732.*pi.*1.333333333333333e-1i)+real(f9.*t562.*t688.*t733.*pi.*1.333333333333333e-1i)+real(f10.*t564.*t689.*t734.*pi.*1.333333333333333e-1i)+real(f11.*t566.*t690.*t735.*pi.*1.333333333333333e-1i)+real(f12.*t568.*t691.*t736.*pi.*1.333333333333333e-1i)+real(f13.*t570.*t692.*t737.*pi.*1.333333333333333e-1i)+real(f14.*t572.*t693.*t738.*pi.*1.333333333333333e-1i)+real(f15.*t574.*t694.*t739.*pi.*1.333333333333333e-1i),real(-(t5.^2.*t23.*t60.*t61.*t411.*t606.*t755.*pi.*(t24-t650))./conj(t140.^a)).*(-2.0./1.5e+1)-real(-(t6.^2.*t23.*t60.*t61.*t413.*t608.*t756.*pi.*(t25-t651))./conj(t141.^a)).*(2.0./1.5e+1)-real(-(t7.^2.*t23.*t60.*t61.*t415.*t610.*t757.*pi.*(t26-t652))./conj(t142.^a)).*(2.0./1.5e+1)-real(-(t8.^2.*t23.*t60.*t61.*t417.*t612.*t758.*pi.*(t27-t653))./conj(t143.^a)).*(2.0./1.5e+1)-real(-(t9.^2.*t23.*t60.*t61.*t419.*t614.*t759.*pi.*(t28-t654))./conj(t144.^a)).*(2.0./1.5e+1)-real(-(t10.^2.*t23.*t60.*t61.*t421.*t616.*t760.*pi.*(t29-t655))./conj(t145.^a)).*(2.0./1.5e+1)-real(-(t11.^2.*t23.*t60.*t61.*t423.*t618.*t761.*pi.*(t30-t656))./conj(t146.^a)).*(2.0./1.5e+1)-real(-(t12.^2.*t23.*t60.*t61.*t425.*t620.*t762.*pi.*(t31-t657))./conj(t147.^a)).*(2.0./1.5e+1)-real(-(t13.^2.*t23.*t60.*t61.*t427.*t622.*t763.*pi.*(t32-t658))./conj(t148.^a)).*(2.0./1.5e+1)-real(-(t14.^2.*t23.*t60.*t61.*t429.*t624.*t764.*pi.*(t33-t659))./conj(t149.^a)).*(2.0./1.5e+1)-real(-(t15.^2.*t23.*t60.*t61.*t431.*t626.*t765.*pi.*(t34-t660))./conj(t150.^a)).*(2.0./1.5e+1)-real(-(t16.^2.*t23.*t60.*t61.*t433.*t628.*t766.*pi.*(t35-t661))./conj(t151.^a)).*(2.0./1.5e+1)-real(-(t17.^2.*t23.*t60.*t61.*t435.*t630.*t767.*pi.*(t36-t662))./conj(t152.^a)).*(2.0./1.5e+1)-real(-(t18.^2.*t23.*t60.*t61.*t437.*t632.*t768.*pi.*(t37-t663))./conj(t153.^a)).*(2.0./1.5e+1)-real(-(t19.^2.*t23.*t60.*t61.*t439.*t634.*t769.*pi.*(t38-t664))./conj(t154.^a)).*(2.0./1.5e+1)+real(f1.^2.*rs.*t55.*t57.*t140.^t54.*t381.*t546.*t680.*t725.*pi).*(2.0./1.5e+1)+real(f2.^2.*rs.*t55.*t57.*t141.^t54.*t383.*t548.*t681.*t726.*pi).*(2.0./1.5e+1)+real(f3.^2.*rs.*t55.*t57.*t142.^t54.*t385.*t550.*t682.*t727.*pi).*(2.0./1.5e+1)+real(f4.^2.*rs.*t55.*t57.*t143.^t54.*t387.*t552.*t683.*t728.*pi).*(2.0./1.5e+1)+real(f5.^2.*rs.*t55.*t57.*t144.^t54.*t389.*t554.*t684.*t729.*pi).*(2.0./1.5e+1)+real(f6.^2.*rs.*t55.*t57.*t145.^t54.*t391.*t556.*t685.*t730.*pi).*(2.0./1.5e+1)+real(f7.^2.*rs.*t55.*t57.*t146.^t54.*t393.*t558.*t686.*t731.*pi).*(2.0./1.5e+1)+real(f8.^2.*rs.*t55.*t57.*t147.^t54.*t395.*t560.*t687.*t732.*pi).*(2.0./1.5e+1)+real(f9.^2.*rs.*t55.*t57.*t148.^t54.*t397.*t562.*t688.*t733.*pi).*(2.0./1.5e+1)+real(f10.^2.*rs.*t55.*t57.*t149.^t54.*t399.*t564.*t689.*t734.*pi).*(2.0./1.5e+1)+real(f11.^2.*rs.*t55.*t57.*t150.^t54.*t401.*t566.*t690.*t735.*pi).*(2.0./1.5e+1)+real(f12.^2.*rs.*t55.*t57.*t151.^t54.*t403.*t568.*t691.*t736.*pi).*(2.0./1.5e+1)+real(f13.^2.*rs.*t55.*t57.*t152.^t54.*t405.*t570.*t692.*t737.*pi).*(2.0./1.5e+1)+real(f14.^2.*rs.*t55.*t57.*t153.^t54.*t407.*t572.*t693.*t738.*pi).*(2.0./1.5e+1)+real(f15.^2.*rs.*t55.*t57.*t154.^t54.*t409.*t574.*t694.*t739.*pi).*(2.0./1.5e+1),real(t546.*t680.*t725)./1.5e+1+real(t548.*t681.*t726)./1.5e+1+real(t550.*t682.*t727)./1.5e+1+real(t552.*t683.*t728)./1.5e+1+real(t554.*t684.*t729)./1.5e+1+real(t556.*t685.*t730)./1.5e+1+real(t558.*t686.*t731)./1.5e+1+real(t560.*t687.*t732)./1.5e+1+real(t562.*t688.*t733)./1.5e+1+real(t564.*t689.*t734)./1.5e+1+real(t566.*t690.*t735)./1.5e+1+real(t568.*t691.*t736)./1.5e+1+real(t570.*t692.*t737)./1.5e+1+real(t572.*t693.*t738)./1.5e+1+real(t574.*t694.*t739)./1.5e+1-real(-t606.*t755.*(t24-t650))./1.5e+1-real(-t608.*t756.*(t25-t651))./1.5e+1-real(-t610.*t757.*(t26-t652))./1.5e+1-real(-t612.*t758.*(t27-t653))./1.5e+1-real(-t614.*t759.*(t28-t654))./1.5e+1-real(-t616.*t760.*(t29-t655))./1.5e+1-real(-t618.*t761.*(t30-t656))./1.5e+1-real(-t620.*t762.*(t31-t657))./1.5e+1-real(-t622.*t763.*(t32-t658))./1.5e+1-real(-t624.*t764.*(t33-t659))./1.5e+1-real(-t626.*t765.*(t34-t660))./1.5e+1-real(-t628.*t766.*(t35-t661))./1.5e+1-real(-t630.*t767.*(t36-t662))./1.5e+1-real(-t632.*t768.*(t37-t663))./1.5e+1-real(-t634.*t769.*(t38-t664))./1.5e+1,real(-t185.*t351.*t755.*conj(t95).*(t24-t650)).*(-1.0./1.5e+1)-real(-t186.*t353.*t756.*conj(t96).*(t25-t651))./1.5e+1-real(-t187.*t355.*t757.*conj(t97).*(t26-t652))./1.5e+1-real(-t188.*t357.*t758.*conj(t98).*(t27-t653))./1.5e+1-real(-t189.*t359.*t759.*conj(t99).*(t28-t654))./1.5e+1-real(-t190.*t361.*t760.*conj(t100).*(t29-t655))./1.5e+1-real(-t191.*t363.*t761.*conj(t101).*(t30-t656))./1.5e+1-real(-t192.*t365.*t762.*conj(t102).*(t31-t657))./1.5e+1-real(-t193.*t367.*t763.*conj(t103).*(t32-t658))./1.5e+1-real(-t194.*t369.*t764.*conj(t104).*(t33-t659))./1.5e+1-real(-t195.*t371.*t765.*conj(t105).*(t34-t660))./1.5e+1-real(-t196.*t373.*t766.*conj(t106).*(t35-t661))./1.5e+1-real(-t197.*t375.*t767.*conj(t107).*(t36-t662))./1.5e+1-real(-t198.*t377.*t768.*conj(t108).*(t37-t663))./1.5e+1-real(-t199.*t379.*t769.*conj(t109).*(t38-t664))./1.5e+1+real(t95.*t155.*t261.*t680.*t725)./1.5e+1+real(t96.*t156.*t263.*t681.*t726)./1.5e+1+real(t97.*t157.*t265.*t682.*t727)./1.5e+1+real(t98.*t158.*t267.*t683.*t728)./1.5e+1+real(t99.*t159.*t269.*t684.*t729)./1.5e+1+real(t100.*t160.*t271.*t685.*t730)./1.5e+1+real(t101.*t161.*t273.*t686.*t731)./1.5e+1+real(t102.*t162.*t275.*t687.*t732)./1.5e+1+real(t103.*t163.*t277.*t688.*t733)./1.5e+1+real(t104.*t164.*t279.*t689.*t734)./1.5e+1+real(t105.*t165.*t281.*t690.*t735)./1.5e+1+real(t106.*t166.*t283.*t691.*t736)./1.5e+1+real(t107.*t167.*t285.*t692.*t737)./1.5e+1+real(t108.*t168.*t287.*t693.*t738)./1.5e+1+real(t109.*t169.*t289.*t694.*t739)./1.5e+1,real(t59.*t261.*t680.*t725).*(-1.0./1.5e+1)-real(t59.*t263.*t681.*t726)./1.5e+1-real(t59.*t265.*t682.*t727)./1.5e+1-real(t59.*t267.*t683.*t728)./1.5e+1-real(t59.*t269.*t684.*t729)./1.5e+1-real(t59.*t271.*t685.*t730)./1.5e+1-real(t59.*t273.*t686.*t731)./1.5e+1-real(t59.*t275.*t687.*t732)./1.5e+1-real(t59.*t277.*t688.*t733)./1.5e+1-real(t59.*t279.*t689.*t734)./1.5e+1-real(t59.*t281.*t690.*t735)./1.5e+1-real(t59.*t283.*t691.*t736)./1.5e+1-real(t59.*t285.*t692.*t737)./1.5e+1-real(t59.*t287.*t693.*t738)./1.5e+1-real(t59.*t289.*t694.*t739)./1.5e+1+real(-t63.*t351.*t755.*(t24-t650))./1.5e+1+real(-t63.*t353.*t756.*(t25-t651))./1.5e+1+real(-t63.*t355.*t757.*(t26-t652))./1.5e+1+real(-t63.*t357.*t758.*(t27-t653))./1.5e+1+real(-t63.*t359.*t759.*(t28-t654))./1.5e+1+real(-t63.*t361.*t760.*(t29-t655))./1.5e+1+real(-t63.*t363.*t761.*(t30-t656))./1.5e+1+real(-t63.*t365.*t762.*(t31-t657))./1.5e+1+real(-t63.*t367.*t763.*(t32-t658))./1.5e+1+real(-t63.*t369.*t764.*(t33-t659))./1.5e+1+real(-t63.*t371.*t765.*(t34-t660))./1.5e+1+real(-t63.*t373.*t766.*(t35-t661))./1.5e+1+real(-t63.*t375.*t767.*(t36-t662))./1.5e+1+real(-t63.*t377.*t768.*(t37-t663))./1.5e+1+real(-t63.*t379.*t769.*(t38-t664))./1.5e+1,real(f1.*t380.*t546.*t680.*t725.*pi.*1.333333333333333e-1i)+real(f2.*t382.*t548.*t681.*t726.*pi.*1.333333333333333e-1i)+real(f3.*t384.*t550.*t682.*t727.*pi.*1.333333333333333e-1i)+real(f4.*t386.*t552.*t683.*t728.*pi.*1.333333333333333e-1i)+real(f5.*t388.*t554.*t684.*t729.*pi.*1.333333333333333e-1i)+real(f6.*t390.*t556.*t685.*t730.*pi.*1.333333333333333e-1i)+real(f7.*t392.*t558.*t686.*t731.*pi.*1.333333333333333e-1i)+real(f8.*t394.*t560.*t687.*t732.*pi.*1.333333333333333e-1i)+real(f9.*t396.*t562.*t688.*t733.*pi.*1.333333333333333e-1i)+real(f10.*t398.*t564.*t689.*t734.*pi.*1.333333333333333e-1i)+real(f11.*t400.*t566.*t690.*t735.*pi.*1.333333333333333e-1i)+real(f12.*t402.*t568.*t691.*t736.*pi.*1.333333333333333e-1i)+real(f13.*t404.*t570.*t692.*t737.*pi.*1.333333333333333e-1i)+real(f14.*t406.*t572.*t693.*t738.*pi.*1.333333333333333e-1i)+real(f15.*t408.*t574.*t694.*t739.*pi.*1.333333333333333e-1i)+real(t5.*t410.*t606.*t755.*pi.*(t24-t650).*-1.333333333333333e-1i)+real(t6.*t412.*t608.*t756.*pi.*(t25-t651).*-1.333333333333333e-1i)+real(t7.*t414.*t610.*t757.*pi.*(t26-t652).*-1.333333333333333e-1i)+real(t8.*t416.*t612.*t758.*pi.*(t27-t653).*-1.333333333333333e-1i)+real(t9.*t418.*t614.*t759.*pi.*(t28-t654).*-1.333333333333333e-1i)+real(t10.*t420.*t616.*t760.*pi.*(t29-t655).*-1.333333333333333e-1i)+real(t11.*t422.*t618.*t761.*pi.*(t30-t656).*-1.333333333333333e-1i)+real(t12.*t424.*t620.*t762.*pi.*(t31-t657).*-1.333333333333333e-1i)+real(t13.*t426.*t622.*t763.*pi.*(t32-t658).*-1.333333333333333e-1i)+real(t14.*t428.*t624.*t764.*pi.*(t33-t659).*-1.333333333333333e-1i)+real(t15.*t430.*t626.*t765.*pi.*(t34-t660).*-1.333333333333333e-1i)+real(t16.*t432.*t628.*t766.*pi.*(t35-t661).*-1.333333333333333e-1i)+real(t17.*t434.*t630.*t767.*pi.*(t36-t662).*-1.333333333333333e-1i)+real(t18.*t436.*t632.*t768.*pi.*(t37-t663).*-1.333333333333333e-1i)+real(t19.*t438.*t634.*t769.*pi.*(t38-t664).*-1.333333333333333e-1i)];