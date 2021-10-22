function grad_RCRC = grad_RCRC(C1,C2,R1,R2,a,b,c,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15)
%GRAD_RCRC
%    GRAD_RCRC = GRAD_RCRC(C1,C2,R1,R2,A,B,C,D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15)

%    This function was generated by the Symbolic Math Toolbox version 8.5.
%    10-Oct-2021 17:56:51

t2 = conj(C1);
t3 = conj(C2);
t4 = log(R1);
t5 = log(R2);
t6 = conj(a);
t7 = conj(b);
t8 = conj(c);
t9 = conj(f1);
t10 = conj(f2);
t11 = conj(f3);
t12 = conj(f4);
t13 = conj(f5);
t14 = conj(f6);
t15 = conj(f7);
t16 = conj(f8);
t17 = conj(f9);
t18 = conj(f10);
t19 = conj(f11);
t20 = conj(f12);
t21 = conj(f13);
t22 = conj(f14);
t23 = conj(f15);
t24 = log(d1);
t25 = log(d2);
t26 = log(d3);
t27 = log(d4);
t28 = log(d5);
t29 = log(d6);
t30 = log(d7);
t31 = log(d8);
t32 = log(d9);
t33 = log(d10);
t34 = log(d11);
t35 = log(d12);
t36 = log(d13);
t37 = log(d14);
t38 = log(d15);
t39 = R1.^a;
t40 = R2.^b;
t60 = 1.0./C1;
t62 = 1.0./C2;
t64 = 1.0./pi;
t65 = a-1.0;
t66 = 1.0./a;
t68 = b-1.0;
t69 = 1.0./b;
t71 = c-1.0;
t72 = 1.0./c;
t74 = 1.0./f1;
t75 = 1.0./f2;
t76 = 1.0./f3;
t77 = 1.0./f4;
t78 = 1.0./f5;
t79 = 1.0./f6;
t80 = 1.0./f7;
t81 = 1.0./f8;
t82 = 1.0./f9;
t83 = 1.0./f10;
t84 = 1.0./f11;
t85 = 1.0./f12;
t86 = 1.0./f13;
t87 = 1.0./f14;
t88 = 1.0./f15;
t41 = conj(t4);
t42 = conj(t5);
t43 = conj(t24);
t44 = conj(t25);
t45 = conj(t26);
t46 = conj(t27);
t47 = conj(t28);
t48 = conj(t29);
t49 = conj(t30);
t50 = conj(t31);
t51 = conj(t32);
t52 = conj(t33);
t53 = conj(t34);
t54 = conj(t35);
t55 = conj(t36);
t56 = conj(t37);
t57 = conj(t38);
t58 = conj(t39);
t59 = conj(t40);
t61 = t60.^2;
t63 = t62.^2;
t67 = t66.^2;
t70 = t69.^2;
t73 = t72.^2;
t89 = 1.0./t2.^2;
t90 = 1.0./t3.^2;
t91 = 1.0./t6;
t93 = 1.0./t7;
t95 = 1.0./t8;
t97 = 1.0./t9;
t98 = 1.0./t10;
t99 = 1.0./t11;
t100 = 1.0./t12;
t101 = 1.0./t13;
t102 = 1.0./t14;
t103 = 1.0./t15;
t104 = 1.0./t16;
t105 = 1.0./t17;
t106 = 1.0./t18;
t107 = 1.0./t19;
t108 = 1.0./t20;
t109 = 1.0./t21;
t110 = 1.0./t22;
t111 = 1.0./t23;
t112 = -t24;
t113 = -t25;
t114 = -t26;
t115 = -t27;
t116 = -t28;
t117 = -t29;
t118 = -t30;
t119 = -t31;
t120 = -t32;
t121 = -t33;
t122 = -t34;
t123 = -t35;
t124 = -t36;
t125 = -t37;
t126 = -t38;
t127 = t4.*t39;
t128 = t5.*t40;
t129 = R1.^t65;
t130 = R2.^t68;
t131 = t66-1.0;
t132 = t69-1.0;
t133 = -t72;
t134 = t72-1.0;
t139 = t60.*t64.*t74.*5.0e-1i;
t140 = t60.*t64.*t75.*5.0e-1i;
t141 = t62.*t64.*t74.*5.0e-1i;
t142 = t60.*t64.*t76.*5.0e-1i;
t143 = t62.*t64.*t75.*5.0e-1i;
t144 = t60.*t64.*t77.*5.0e-1i;
t145 = t62.*t64.*t76.*5.0e-1i;
t146 = t60.*t64.*t78.*5.0e-1i;
t147 = t62.*t64.*t77.*5.0e-1i;
t148 = t60.*t64.*t79.*5.0e-1i;
t149 = t62.*t64.*t78.*5.0e-1i;
t150 = t60.*t64.*t80.*5.0e-1i;
t151 = t62.*t64.*t79.*5.0e-1i;
t152 = t60.*t64.*t81.*5.0e-1i;
t153 = t62.*t64.*t80.*5.0e-1i;
t154 = t60.*t64.*t82.*5.0e-1i;
t155 = t62.*t64.*t81.*5.0e-1i;
t156 = t60.*t64.*t83.*5.0e-1i;
t157 = t62.*t64.*t82.*5.0e-1i;
t158 = t60.*t64.*t84.*5.0e-1i;
t159 = t62.*t64.*t83.*5.0e-1i;
t160 = t60.*t64.*t85.*5.0e-1i;
t161 = t62.*t64.*t84.*5.0e-1i;
t162 = t60.*t64.*t86.*5.0e-1i;
t163 = t62.*t64.*t85.*5.0e-1i;
t164 = t60.*t64.*t87.*5.0e-1i;
t165 = t62.*t64.*t86.*5.0e-1i;
t166 = t60.*t64.*t88.*5.0e-1i;
t167 = t62.*t64.*t87.*5.0e-1i;
t168 = t62.*t64.*t88.*5.0e-1i;
t92 = t91.^2;
t94 = t93.^2;
t96 = t95.^2;
t135 = conj(t129);
t136 = conj(t130);
t137 = t41.*t58;
t138 = t42.*t59;
t169 = -t139;
t170 = -t140;
t171 = -t141;
t172 = -t142;
t173 = -t143;
t174 = -t144;
t175 = -t145;
t176 = -t146;
t177 = -t147;
t178 = -t148;
t179 = -t149;
t180 = -t150;
t181 = -t151;
t182 = -t152;
t183 = -t153;
t184 = -t154;
t185 = -t155;
t186 = -t156;
t187 = -t157;
t188 = -t158;
t189 = -t159;
t190 = -t160;
t191 = -t161;
t192 = -t162;
t193 = -t163;
t194 = -t164;
t195 = -t165;
t196 = -t166;
t197 = -t167;
t198 = -t168;
t199 = log(t169);
t200 = log(t170);
t201 = log(t171);
t202 = log(t172);
t203 = log(t173);
t204 = log(t174);
t205 = log(t175);
t206 = log(t176);
t207 = log(t177);
t208 = log(t178);
t209 = log(t179);
t210 = log(t180);
t211 = log(t181);
t212 = log(t182);
t213 = log(t183);
t214 = log(t184);
t215 = log(t185);
t216 = log(t186);
t217 = log(t187);
t218 = log(t188);
t219 = log(t189);
t220 = log(t190);
t221 = log(t191);
t222 = log(t192);
t223 = log(t193);
t224 = log(t194);
t225 = log(t195);
t226 = log(t196);
t227 = log(t197);
t228 = log(t198);
t229 = t169.^a;
t230 = t170.^a;
t231 = t172.^a;
t232 = t174.^a;
t233 = t176.^a;
t234 = t178.^a;
t235 = t180.^a;
t236 = t182.^a;
t237 = t184.^a;
t238 = t186.^a;
t239 = t188.^a;
t240 = t190.^a;
t241 = t192.^a;
t242 = t194.^a;
t243 = t196.^a;
t244 = t171.^b;
t245 = t173.^b;
t246 = t175.^b;
t247 = t177.^b;
t248 = t179.^b;
t249 = t181.^b;
t250 = t183.^b;
t251 = t185.^b;
t252 = t187.^b;
t253 = t189.^b;
t254 = t191.^b;
t255 = t193.^b;
t256 = t195.^b;
t257 = t197.^b;
t258 = t198.^b;
t259 = t169.^t65;
t260 = t170.^t65;
t261 = t172.^t65;
t262 = t174.^t65;
t263 = t176.^t65;
t264 = t178.^t65;
t265 = t180.^t65;
t266 = t182.^t65;
t267 = t184.^t65;
t268 = t186.^t65;
t269 = t188.^t65;
t270 = t190.^t65;
t271 = t192.^t65;
t272 = t194.^t65;
t273 = t196.^t65;
t274 = t171.^t68;
t275 = t173.^t68;
t276 = t175.^t68;
t277 = t177.^t68;
t278 = t179.^t68;
t279 = t181.^t68;
t280 = t183.^t68;
t281 = t185.^t68;
t282 = t187.^t68;
t283 = t189.^t68;
t284 = t191.^t68;
t285 = t193.^t68;
t286 = t195.^t68;
t287 = t197.^t68;
t288 = t198.^t68;
t289 = t39+t229;
t290 = t39+t230;
t291 = t39+t231;
t292 = t39+t232;
t293 = t39+t233;
t294 = t39+t234;
t295 = t39+t235;
t296 = t39+t236;
t297 = t39+t237;
t298 = t39+t238;
t299 = t39+t239;
t300 = t39+t240;
t301 = t39+t241;
t302 = t39+t242;
t303 = t39+t243;
t304 = t40+t244;
t305 = t40+t245;
t306 = t40+t246;
t307 = t40+t247;
t308 = t40+t248;
t309 = t40+t249;
t310 = t40+t250;
t311 = t40+t251;
t312 = t40+t252;
t313 = t40+t253;
t314 = t40+t254;
t315 = t40+t255;
t316 = t40+t256;
t317 = t40+t257;
t318 = t40+t258;
t319 = log(t289);
t320 = log(t290);
t321 = log(t291);
t322 = log(t292);
t323 = log(t293);
t324 = log(t294);
t325 = log(t295);
t326 = log(t296);
t327 = log(t297);
t328 = log(t298);
t329 = log(t299);
t330 = log(t300);
t331 = log(t301);
t332 = log(t302);
t333 = log(t303);
t334 = log(t304);
t335 = log(t305);
t336 = log(t306);
t337 = log(t307);
t338 = log(t308);
t339 = log(t309);
t340 = log(t310);
t341 = log(t311);
t342 = log(t312);
t343 = log(t313);
t344 = log(t314);
t345 = log(t315);
t346 = log(t316);
t347 = log(t317);
t348 = log(t318);
t349 = t289.^t66;
t350 = t290.^t66;
t351 = t291.^t66;
t352 = t292.^t66;
t353 = t293.^t66;
t354 = t294.^t66;
t355 = t295.^t66;
t356 = t296.^t66;
t357 = t297.^t66;
t358 = t298.^t66;
t359 = t299.^t66;
t360 = t300.^t66;
t361 = t301.^t66;
t362 = t302.^t66;
t363 = t303.^t66;
t364 = t304.^t69;
t365 = t305.^t69;
t366 = t306.^t69;
t367 = t307.^t69;
t368 = t308.^t69;
t369 = t309.^t69;
t370 = t310.^t69;
t371 = t311.^t69;
t372 = t312.^t69;
t373 = t313.^t69;
t374 = t314.^t69;
t375 = t315.^t69;
t376 = t316.^t69;
t377 = t317.^t69;
t378 = t318.^t69;
t439 = t289.^t131;
t440 = t290.^t131;
t441 = t291.^t131;
t442 = t292.^t131;
t443 = t293.^t131;
t444 = t294.^t131;
t445 = t295.^t131;
t446 = t296.^t131;
t447 = t297.^t131;
t448 = t298.^t131;
t449 = t299.^t131;
t450 = t300.^t131;
t451 = t301.^t131;
t452 = t302.^t131;
t453 = t303.^t131;
t454 = t304.^t132;
t455 = t305.^t132;
t456 = t306.^t132;
t457 = t307.^t132;
t458 = t308.^t132;
t459 = t309.^t132;
t460 = t310.^t132;
t461 = t311.^t132;
t462 = t312.^t132;
t463 = t313.^t132;
t464 = t314.^t132;
t465 = t315.^t132;
t466 = t316.^t132;
t467 = t317.^t132;
t468 = t318.^t132;
t379 = log(t349);
t380 = log(t350);
t381 = log(t351);
t382 = log(t352);
t383 = log(t353);
t384 = log(t354);
t385 = log(t355);
t386 = log(t356);
t387 = log(t357);
t388 = log(t358);
t389 = log(t359);
t390 = log(t360);
t391 = log(t361);
t392 = log(t362);
t393 = log(t363);
t394 = log(t364);
t395 = log(t365);
t396 = log(t366);
t397 = log(t367);
t398 = log(t368);
t399 = log(t369);
t400 = log(t370);
t401 = log(t371);
t402 = log(t372);
t403 = log(t373);
t404 = log(t374);
t405 = log(t375);
t406 = log(t376);
t407 = log(t377);
t408 = log(t378);
t409 = t349.^c;
t410 = t350.^c;
t411 = t351.^c;
t412 = t352.^c;
t413 = t353.^c;
t414 = t354.^c;
t415 = t355.^c;
t416 = t356.^c;
t417 = t357.^c;
t418 = t358.^c;
t419 = t359.^c;
t420 = t360.^c;
t421 = t361.^c;
t422 = t362.^c;
t423 = t363.^c;
t424 = t364.^c;
t425 = t365.^c;
t426 = t366.^c;
t427 = t367.^c;
t428 = t368.^c;
t429 = t369.^c;
t430 = t370.^c;
t431 = t371.^c;
t432 = t372.^c;
t433 = t373.^c;
t434 = t374.^c;
t435 = t375.^c;
t436 = t376.^c;
t437 = t377.^c;
t438 = t378.^c;
t469 = conj(t439);
t470 = conj(t440);
t471 = conj(t441);
t472 = conj(t442);
t473 = conj(t443);
t474 = conj(t444);
t475 = conj(t445);
t476 = conj(t446);
t477 = conj(t447);
t478 = conj(t448);
t479 = conj(t449);
t480 = conj(t450);
t481 = conj(t451);
t482 = conj(t452);
t483 = conj(t453);
t484 = conj(t454);
t485 = conj(t455);
t486 = conj(t456);
t487 = conj(t457);
t488 = conj(t458);
t489 = conj(t459);
t490 = conj(t460);
t491 = conj(t461);
t492 = conj(t462);
t493 = conj(t463);
t494 = conj(t464);
t495 = conj(t465);
t496 = conj(t466);
t497 = conj(t467);
t498 = conj(t468);
t499 = t349.^t71;
t500 = t350.^t71;
t501 = t351.^t71;
t502 = t352.^t71;
t503 = t353.^t71;
t504 = t354.^t71;
t505 = t355.^t71;
t506 = t356.^t71;
t507 = t357.^t71;
t508 = t358.^t71;
t509 = t359.^t71;
t510 = t360.^t71;
t511 = t361.^t71;
t512 = t362.^t71;
t513 = t363.^t71;
t514 = t364.^t71;
t515 = t365.^t71;
t516 = t366.^t71;
t517 = t367.^t71;
t518 = t368.^t71;
t519 = t369.^t71;
t520 = t370.^t71;
t521 = t371.^t71;
t522 = t372.^t71;
t523 = t373.^t71;
t524 = t374.^t71;
t525 = t375.^t71;
t526 = t376.^t71;
t527 = t377.^t71;
t528 = t378.^t71;
t529 = conj(t499);
t530 = conj(t500);
t531 = conj(t501);
t532 = conj(t502);
t533 = conj(t503);
t534 = conj(t504);
t535 = conj(t505);
t536 = conj(t506);
t537 = conj(t507);
t538 = conj(t508);
t539 = conj(t509);
t540 = conj(t510);
t541 = conj(t511);
t542 = conj(t512);
t543 = conj(t513);
t544 = conj(t514);
t545 = conj(t515);
t546 = conj(t516);
t547 = conj(t517);
t548 = conj(t518);
t549 = conj(t519);
t550 = conj(t520);
t551 = conj(t521);
t552 = conj(t522);
t553 = conj(t523);
t554 = conj(t524);
t555 = conj(t525);
t556 = conj(t526);
t557 = conj(t527);
t558 = conj(t528);
t559 = t409+t424;
t560 = t410+t425;
t561 = t411+t426;
t562 = t412+t427;
t563 = t413+t428;
t564 = t414+t429;
t565 = t415+t430;
t566 = t416+t431;
t567 = t417+t432;
t568 = t418+t433;
t569 = t419+t434;
t570 = t420+t435;
t571 = t421+t436;
t572 = t422+t437;
t573 = t423+t438;
t574 = log(t559);
t575 = log(t560);
t576 = log(t561);
t577 = log(t562);
t578 = log(t563);
t579 = log(t564);
t580 = log(t565);
t581 = log(t566);
t582 = log(t567);
t583 = log(t568);
t584 = log(t569);
t585 = log(t570);
t586 = log(t571);
t587 = log(t572);
t588 = log(t573);
t589 = t559.^t72;
t590 = t560.^t72;
t591 = t561.^t72;
t592 = t562.^t72;
t593 = t563.^t72;
t594 = t564.^t72;
t595 = t565.^t72;
t596 = t566.^t72;
t597 = t567.^t72;
t598 = t568.^t72;
t599 = t569.^t72;
t600 = t570.^t72;
t601 = t571.^t72;
t602 = t572.^t72;
t603 = t573.^t72;
t649 = t559.^t133;
t650 = t560.^t133;
t651 = t561.^t133;
t652 = t562.^t133;
t653 = t563.^t133;
t654 = t564.^t133;
t655 = t565.^t133;
t656 = t566.^t133;
t657 = t567.^t133;
t658 = t568.^t133;
t659 = t569.^t133;
t660 = t570.^t133;
t661 = t571.^t133;
t662 = t572.^t133;
t663 = t573.^t133;
t664 = t559.^t134;
t665 = t560.^t134;
t666 = t561.^t134;
t667 = t562.^t134;
t668 = t563.^t134;
t669 = t564.^t134;
t670 = t565.^t134;
t671 = t566.^t134;
t672 = t567.^t134;
t673 = t568.^t134;
t674 = t569.^t134;
t675 = t570.^t134;
t676 = t571.^t134;
t677 = t572.^t134;
t678 = t573.^t134;
t604 = log(t589);
t605 = log(t590);
t606 = log(t591);
t607 = log(t592);
t608 = log(t593);
t609 = log(t594);
t610 = log(t595);
t611 = log(t596);
t612 = log(t597);
t613 = log(t598);
t614 = log(t599);
t615 = log(t600);
t616 = log(t601);
t617 = log(t602);
t618 = log(t603);
t619 = conj(t589);
t620 = conj(t590);
t621 = conj(t591);
t622 = conj(t592);
t623 = conj(t593);
t624 = conj(t594);
t625 = conj(t595);
t626 = conj(t596);
t627 = conj(t597);
t628 = conj(t598);
t629 = conj(t599);
t630 = conj(t600);
t631 = conj(t601);
t632 = conj(t602);
t633 = conj(t603);
t679 = conj(t664);
t680 = conj(t665);
t681 = conj(t666);
t682 = conj(t667);
t683 = conj(t668);
t684 = conj(t669);
t685 = conj(t670);
t686 = conj(t671);
t687 = conj(t672);
t688 = conj(t673);
t689 = conj(t674);
t690 = conj(t675);
t691 = conj(t676);
t692 = conj(t677);
t693 = conj(t678);
t634 = conj(t604);
t635 = conj(t605);
t636 = conj(t606);
t637 = conj(t607);
t638 = conj(t608);
t639 = conj(t609);
t640 = conj(t610);
t641 = conj(t611);
t642 = conj(t612);
t643 = conj(t613);
t644 = conj(t614);
t645 = conj(t615);
t646 = conj(t616);
t647 = conj(t617);
t648 = conj(t618);
t694 = 1.0./t619;
t695 = 1.0./t620;
t696 = 1.0./t621;
t697 = 1.0./t622;
t698 = 1.0./t623;
t699 = 1.0./t624;
t700 = 1.0./t625;
t701 = 1.0./t626;
t702 = 1.0./t627;
t703 = 1.0./t628;
t704 = 1.0./t629;
t705 = 1.0./t630;
t706 = 1.0./t631;
t707 = 1.0./t632;
t708 = 1.0./t633;
t724 = t112+t604;
t725 = t113+t605;
t726 = t114+t606;
t727 = t115+t607;
t728 = t116+t608;
t729 = t117+t609;
t730 = t118+t610;
t731 = t119+t611;
t732 = t120+t612;
t733 = t121+t613;
t734 = t122+t614;
t735 = t123+t615;
t736 = t124+t616;
t737 = t125+t617;
t738 = t126+t618;
t709 = -t634;
t710 = -t635;
t711 = -t636;
t712 = -t637;
t713 = -t638;
t714 = -t639;
t715 = -t640;
t716 = -t641;
t717 = -t642;
t718 = -t643;
t719 = -t644;
t720 = -t645;
t721 = -t646;
t722 = -t647;
t723 = -t648;
t739 = t43+t709;
t740 = t44+t710;
t741 = t45+t711;
t742 = t46+t712;
t743 = t47+t713;
t744 = t48+t714;
t745 = t49+t715;
t746 = t50+t716;
t747 = t51+t717;
t748 = t52+t718;
t749 = t53+t719;
t750 = t54+t720;
t751 = t55+t721;
t752 = t56+t722;
t753 = t57+t723;
grad_RCRC = [real(t64.*t89.*t97.*t469.*t529.*t679.*t694.*conj(t259).*(t24-t604).*3.333333333333333e-2i)+real(t64.*t89.*t98.*t470.*t530.*t680.*t695.*conj(t260).*(t25-t605).*3.333333333333333e-2i)+real(t64.*t89.*t99.*t471.*t531.*t681.*t696.*conj(t261).*(t26-t606).*3.333333333333333e-2i)+real(t64.*t89.*t100.*t472.*t532.*t682.*t697.*conj(t262).*(t27-t607).*3.333333333333333e-2i)+real(t64.*t89.*t101.*t473.*t533.*t683.*t698.*conj(t263).*(t28-t608).*3.333333333333333e-2i)+real(t64.*t89.*t102.*t474.*t534.*t684.*t699.*conj(t264).*(t29-t609).*3.333333333333333e-2i)+real(t64.*t89.*t103.*t475.*t535.*t685.*t700.*conj(t265).*(t30-t610).*3.333333333333333e-2i)+real(t64.*t89.*t104.*t476.*t536.*t686.*t701.*conj(t266).*(t31-t611).*3.333333333333333e-2i)+real(t64.*t89.*t105.*t477.*t537.*t687.*t702.*conj(t267).*(t32-t612).*3.333333333333333e-2i)+real(t64.*t89.*t106.*t478.*t538.*t688.*t703.*conj(t268).*(t33-t613).*3.333333333333333e-2i)+real(t64.*t89.*t107.*t479.*t539.*t689.*t704.*conj(t269).*(t34-t614).*3.333333333333333e-2i)+real(t64.*t89.*t108.*t480.*t540.*t690.*t705.*conj(t270).*(t35-t615).*3.333333333333333e-2i)+real(t64.*t89.*t109.*t481.*t541.*t691.*t706.*conj(t271).*(t36-t616).*3.333333333333333e-2i)+real(t64.*t89.*t110.*t482.*t542.*t692.*t707.*conj(t272).*(t37-t617).*3.333333333333333e-2i)+real(t64.*t89.*t111.*t483.*t543.*t693.*t708.*conj(t273).*(t38-t618).*3.333333333333333e-2i)+real(t61.*t64.*t74.*t259.*t439.*t499.*t649.*t664.*t739.*-3.333333333333333e-2i)+real(t61.*t64.*t75.*t260.*t440.*t500.*t650.*t665.*t740.*-3.333333333333333e-2i)+real(t61.*t64.*t76.*t261.*t441.*t501.*t651.*t666.*t741.*-3.333333333333333e-2i)+real(t61.*t64.*t77.*t262.*t442.*t502.*t652.*t667.*t742.*-3.333333333333333e-2i)+real(t61.*t64.*t78.*t263.*t443.*t503.*t653.*t668.*t743.*-3.333333333333333e-2i)+real(t61.*t64.*t79.*t264.*t444.*t504.*t654.*t669.*t744.*-3.333333333333333e-2i)+real(t61.*t64.*t80.*t265.*t445.*t505.*t655.*t670.*t745.*-3.333333333333333e-2i)+real(t61.*t64.*t81.*t266.*t446.*t506.*t656.*t671.*t746.*-3.333333333333333e-2i)+real(t61.*t64.*t82.*t267.*t447.*t507.*t657.*t672.*t747.*-3.333333333333333e-2i)+real(t61.*t64.*t83.*t268.*t448.*t508.*t658.*t673.*t748.*-3.333333333333333e-2i)+real(t61.*t64.*t84.*t269.*t449.*t509.*t659.*t674.*t749.*-3.333333333333333e-2i)+real(t61.*t64.*t85.*t270.*t450.*t510.*t660.*t675.*t750.*-3.333333333333333e-2i)+real(t61.*t64.*t86.*t271.*t451.*t511.*t661.*t676.*t751.*-3.333333333333333e-2i)+real(t61.*t64.*t87.*t272.*t452.*t512.*t662.*t677.*t752.*-3.333333333333333e-2i)+real(t61.*t64.*t88.*t273.*t453.*t513.*t663.*t678.*t753.*-3.333333333333333e-2i);real(t64.*t90.*t97.*t484.*t544.*t679.*t694.*conj(t274).*(t24-t604).*3.333333333333333e-2i)+real(t64.*t90.*t98.*t485.*t545.*t680.*t695.*conj(t275).*(t25-t605).*3.333333333333333e-2i)+real(t64.*t90.*t99.*t486.*t546.*t681.*t696.*conj(t276).*(t26-t606).*3.333333333333333e-2i)+real(t64.*t90.*t100.*t487.*t547.*t682.*t697.*conj(t277).*(t27-t607).*3.333333333333333e-2i)+real(t64.*t90.*t101.*t488.*t548.*t683.*t698.*conj(t278).*(t28-t608).*3.333333333333333e-2i)+real(t64.*t90.*t102.*t489.*t549.*t684.*t699.*conj(t279).*(t29-t609).*3.333333333333333e-2i)+real(t64.*t90.*t103.*t490.*t550.*t685.*t700.*conj(t280).*(t30-t610).*3.333333333333333e-2i)+real(t64.*t90.*t104.*t491.*t551.*t686.*t701.*conj(t281).*(t31-t611).*3.333333333333333e-2i)+real(t64.*t90.*t105.*t492.*t552.*t687.*t702.*conj(t282).*(t32-t612).*3.333333333333333e-2i)+real(t64.*t90.*t106.*t493.*t553.*t688.*t703.*conj(t283).*(t33-t613).*3.333333333333333e-2i)+real(t64.*t90.*t107.*t494.*t554.*t689.*t704.*conj(t284).*(t34-t614).*3.333333333333333e-2i)+real(t64.*t90.*t108.*t495.*t555.*t690.*t705.*conj(t285).*(t35-t615).*3.333333333333333e-2i)+real(t64.*t90.*t109.*t496.*t556.*t691.*t706.*conj(t286).*(t36-t616).*3.333333333333333e-2i)+real(t64.*t90.*t110.*t497.*t557.*t692.*t707.*conj(t287).*(t37-t617).*3.333333333333333e-2i)+real(t64.*t90.*t111.*t498.*t558.*t693.*t708.*conj(t288).*(t38-t618).*3.333333333333333e-2i)+real(t63.*t64.*t74.*t274.*t454.*t514.*t649.*t664.*t739.*-3.333333333333333e-2i)+real(t63.*t64.*t75.*t275.*t455.*t515.*t650.*t665.*t740.*-3.333333333333333e-2i)+real(t63.*t64.*t76.*t276.*t456.*t516.*t651.*t666.*t741.*-3.333333333333333e-2i)+real(t63.*t64.*t77.*t277.*t457.*t517.*t652.*t667.*t742.*-3.333333333333333e-2i)+real(t63.*t64.*t78.*t278.*t458.*t518.*t653.*t668.*t743.*-3.333333333333333e-2i)+real(t63.*t64.*t79.*t279.*t459.*t519.*t654.*t669.*t744.*-3.333333333333333e-2i)+real(t63.*t64.*t80.*t280.*t460.*t520.*t655.*t670.*t745.*-3.333333333333333e-2i)+real(t63.*t64.*t81.*t281.*t461.*t521.*t656.*t671.*t746.*-3.333333333333333e-2i)+real(t63.*t64.*t82.*t282.*t462.*t522.*t657.*t672.*t747.*-3.333333333333333e-2i)+real(t63.*t64.*t83.*t283.*t463.*t523.*t658.*t673.*t748.*-3.333333333333333e-2i)+real(t63.*t64.*t84.*t284.*t464.*t524.*t659.*t674.*t749.*-3.333333333333333e-2i)+real(t63.*t64.*t85.*t285.*t465.*t525.*t660.*t675.*t750.*-3.333333333333333e-2i)+real(t63.*t64.*t86.*t286.*t466.*t526.*t661.*t676.*t751.*-3.333333333333333e-2i)+real(t63.*t64.*t87.*t287.*t467.*t527.*t662.*t677.*t752.*-3.333333333333333e-2i)+real(t63.*t64.*t88.*t288.*t468.*t528.*t663.*t678.*t753.*-3.333333333333333e-2i);real(t129.*t439.*t499.*t649.*t664.*t739).*(-1.0./1.5e+1)-real(t129.*t440.*t500.*t650.*t665.*t740)./1.5e+1-real(t129.*t441.*t501.*t651.*t666.*t741)./1.5e+1-real(t129.*t442.*t502.*t652.*t667.*t742)./1.5e+1-real(t129.*t443.*t503.*t653.*t668.*t743)./1.5e+1-real(t129.*t444.*t504.*t654.*t669.*t744)./1.5e+1-real(t129.*t445.*t505.*t655.*t670.*t745)./1.5e+1-real(t129.*t446.*t506.*t656.*t671.*t746)./1.5e+1-real(t129.*t447.*t507.*t657.*t672.*t747)./1.5e+1-real(t129.*t448.*t508.*t658.*t673.*t748)./1.5e+1-real(t129.*t449.*t509.*t659.*t674.*t749)./1.5e+1-real(t129.*t450.*t510.*t660.*t675.*t750)./1.5e+1-real(t129.*t451.*t511.*t661.*t676.*t751)./1.5e+1-real(t129.*t452.*t512.*t662.*t677.*t752)./1.5e+1-real(t129.*t453.*t513.*t663.*t678.*t753)./1.5e+1+real(-t135.*t469.*t529.*t679.*t694.*(t24-t604))./1.5e+1+real(-t135.*t470.*t530.*t680.*t695.*(t25-t605))./1.5e+1+real(-t135.*t471.*t531.*t681.*t696.*(t26-t606))./1.5e+1+real(-t135.*t472.*t532.*t682.*t697.*(t27-t607))./1.5e+1+real(-t135.*t473.*t533.*t683.*t698.*(t28-t608))./1.5e+1+real(-t135.*t474.*t534.*t684.*t699.*(t29-t609))./1.5e+1+real(-t135.*t475.*t535.*t685.*t700.*(t30-t610))./1.5e+1+real(-t135.*t476.*t536.*t686.*t701.*(t31-t611))./1.5e+1+real(-t135.*t477.*t537.*t687.*t702.*(t32-t612))./1.5e+1+real(-t135.*t478.*t538.*t688.*t703.*(t33-t613))./1.5e+1+real(-t135.*t479.*t539.*t689.*t704.*(t34-t614))./1.5e+1+real(-t135.*t480.*t540.*t690.*t705.*(t35-t615))./1.5e+1+real(-t135.*t481.*t541.*t691.*t706.*(t36-t616))./1.5e+1+real(-t135.*t482.*t542.*t692.*t707.*(t37-t617))./1.5e+1+real(-t135.*t483.*t543.*t693.*t708.*(t38-t618))./1.5e+1;real(t130.*t454.*t514.*t649.*t664.*t739).*(-1.0./1.5e+1)-real(t130.*t455.*t515.*t650.*t665.*t740)./1.5e+1-real(t130.*t456.*t516.*t651.*t666.*t741)./1.5e+1-real(t130.*t457.*t517.*t652.*t667.*t742)./1.5e+1-real(t130.*t458.*t518.*t653.*t668.*t743)./1.5e+1-real(t130.*t459.*t519.*t654.*t669.*t744)./1.5e+1-real(t130.*t460.*t520.*t655.*t670.*t745)./1.5e+1-real(t130.*t461.*t521.*t656.*t671.*t746)./1.5e+1-real(t130.*t462.*t522.*t657.*t672.*t747)./1.5e+1-real(t130.*t463.*t523.*t658.*t673.*t748)./1.5e+1-real(t130.*t464.*t524.*t659.*t674.*t749)./1.5e+1-real(t130.*t465.*t525.*t660.*t675.*t750)./1.5e+1-real(t130.*t466.*t526.*t661.*t676.*t751)./1.5e+1-real(t130.*t467.*t527.*t662.*t677.*t752)./1.5e+1-real(t130.*t468.*t528.*t663.*t678.*t753)./1.5e+1+real(-t136.*t484.*t544.*t679.*t694.*(t24-t604))./1.5e+1+real(-t136.*t485.*t545.*t680.*t695.*(t25-t605))./1.5e+1+real(-t136.*t486.*t546.*t681.*t696.*(t26-t606))./1.5e+1+real(-t136.*t487.*t547.*t682.*t697.*(t27-t607))./1.5e+1+real(-t136.*t488.*t548.*t683.*t698.*(t28-t608))./1.5e+1+real(-t136.*t489.*t549.*t684.*t699.*(t29-t609))./1.5e+1+real(-t136.*t490.*t550.*t685.*t700.*(t30-t610))./1.5e+1+real(-t136.*t491.*t551.*t686.*t701.*(t31-t611))./1.5e+1+real(-t136.*t492.*t552.*t687.*t702.*(t32-t612))./1.5e+1+real(-t136.*t493.*t553.*t688.*t703.*(t33-t613))./1.5e+1+real(-t136.*t494.*t554.*t689.*t704.*(t34-t614))./1.5e+1+real(-t136.*t495.*t555.*t690.*t705.*(t35-t615))./1.5e+1+real(-t136.*t496.*t556.*t691.*t706.*(t36-t616))./1.5e+1+real(-t136.*t497.*t557.*t692.*t707.*(t37-t617))./1.5e+1+real(-t136.*t498.*t558.*t693.*t708.*(t38-t618))./1.5e+1;real(t529.*t679.*t694.*(t91.*t469.*(t137+conj(t199).*conj(t229))-t92.*conj(t319).*conj(t349)).*(t24-t604)).*(-1.0./1.5e+1)-real(t530.*t680.*t695.*(t91.*t470.*(t137+conj(t200).*conj(t230))-t92.*conj(t320).*conj(t350)).*(t25-t605))./1.5e+1-real(t531.*t681.*t696.*(t91.*t471.*(t137+conj(t202).*conj(t231))-t92.*conj(t321).*conj(t351)).*(t26-t606))./1.5e+1-real(t532.*t682.*t697.*(t91.*t472.*(t137+conj(t204).*conj(t232))-t92.*conj(t322).*conj(t352)).*(t27-t607))./1.5e+1-real(t533.*t683.*t698.*(t91.*t473.*(t137+conj(t206).*conj(t233))-t92.*conj(t323).*conj(t353)).*(t28-t608))./1.5e+1-real(t534.*t684.*t699.*(t91.*t474.*(t137+conj(t208).*conj(t234))-t92.*conj(t324).*conj(t354)).*(t29-t609))./1.5e+1-real(t535.*t685.*t700.*(t91.*t475.*(t137+conj(t210).*conj(t235))-t92.*conj(t325).*conj(t355)).*(t30-t610))./1.5e+1-real(t536.*t686.*t701.*(t91.*t476.*(t137+conj(t212).*conj(t236))-t92.*conj(t326).*conj(t356)).*(t31-t611))./1.5e+1-real(t537.*t687.*t702.*(t91.*t477.*(t137+conj(t214).*conj(t237))-t92.*conj(t327).*conj(t357)).*(t32-t612))./1.5e+1-real(t538.*t688.*t703.*(t91.*t478.*(t137+conj(t216).*conj(t238))-t92.*conj(t328).*conj(t358)).*(t33-t613))./1.5e+1-real(t539.*t689.*t704.*(t91.*t479.*(t137+conj(t218).*conj(t239))-t92.*conj(t329).*conj(t359)).*(t34-t614))./1.5e+1-real(t540.*t690.*t705.*(t91.*t480.*(t137+conj(t220).*conj(t240))-t92.*conj(t330).*conj(t360)).*(t35-t615))./1.5e+1-real(t541.*t691.*t706.*(t91.*t481.*(t137+conj(t222).*conj(t241))-t92.*conj(t331).*conj(t361)).*(t36-t616))./1.5e+1-real(t542.*t692.*t707.*(t91.*t482.*(t137+conj(t224).*conj(t242))-t92.*conj(t332).*conj(t362)).*(t37-t617))./1.5e+1-real(t543.*t693.*t708.*(t91.*t483.*(t137+conj(t226).*conj(t243))-t92.*conj(t333).*conj(t363)).*(t38-t618))./1.5e+1-real(t499.*t649.*t664.*t739.*(t66.*t439.*(t127+t199.*t229)-t67.*t319.*t349))./1.5e+1-real(t500.*t650.*t665.*t740.*(t66.*t440.*(t127+t200.*t230)-t67.*t320.*t350))./1.5e+1-real(t501.*t651.*t666.*t741.*(t66.*t441.*(t127+t202.*t231)-t67.*t321.*t351))./1.5e+1-real(t502.*t652.*t667.*t742.*(t66.*t442.*(t127+t204.*t232)-t67.*t322.*t352))./1.5e+1-real(t503.*t653.*t668.*t743.*(t66.*t443.*(t127+t206.*t233)-t67.*t323.*t353))./1.5e+1-real(t504.*t654.*t669.*t744.*(t66.*t444.*(t127+t208.*t234)-t67.*t324.*t354))./1.5e+1-real(t505.*t655.*t670.*t745.*(t66.*t445.*(t127+t210.*t235)-t67.*t325.*t355))./1.5e+1-real(t506.*t656.*t671.*t746.*(t66.*t446.*(t127+t212.*t236)-t67.*t326.*t356))./1.5e+1-real(t507.*t657.*t672.*t747.*(t66.*t447.*(t127+t214.*t237)-t67.*t327.*t357))./1.5e+1-real(t508.*t658.*t673.*t748.*(t66.*t448.*(t127+t216.*t238)-t67.*t328.*t358))./1.5e+1-real(t509.*t659.*t674.*t749.*(t66.*t449.*(t127+t218.*t239)-t67.*t329.*t359))./1.5e+1-real(t510.*t660.*t675.*t750.*(t66.*t450.*(t127+t220.*t240)-t67.*t330.*t360))./1.5e+1-real(t511.*t661.*t676.*t751.*(t66.*t451.*(t127+t222.*t241)-t67.*t331.*t361))./1.5e+1-real(t512.*t662.*t677.*t752.*(t66.*t452.*(t127+t224.*t242)-t67.*t332.*t362))./1.5e+1-real(t513.*t663.*t678.*t753.*(t66.*t453.*(t127+t226.*t243)-t67.*t333.*t363))./1.5e+1;real(t544.*t679.*t694.*(t93.*t484.*(t138+conj(t201).*conj(t244))-t94.*conj(t334).*conj(t364)).*(t24-t604)).*(-1.0./1.5e+1)-real(t545.*t680.*t695.*(t93.*t485.*(t138+conj(t203).*conj(t245))-t94.*conj(t335).*conj(t365)).*(t25-t605))./1.5e+1-real(t546.*t681.*t696.*(t93.*t486.*(t138+conj(t205).*conj(t246))-t94.*conj(t336).*conj(t366)).*(t26-t606))./1.5e+1-real(t547.*t682.*t697.*(t93.*t487.*(t138+conj(t207).*conj(t247))-t94.*conj(t337).*conj(t367)).*(t27-t607))./1.5e+1-real(t548.*t683.*t698.*(t93.*t488.*(t138+conj(t209).*conj(t248))-t94.*conj(t338).*conj(t368)).*(t28-t608))./1.5e+1-real(t549.*t684.*t699.*(t93.*t489.*(t138+conj(t211).*conj(t249))-t94.*conj(t339).*conj(t369)).*(t29-t609))./1.5e+1-real(t550.*t685.*t700.*(t93.*t490.*(t138+conj(t213).*conj(t250))-t94.*conj(t340).*conj(t370)).*(t30-t610))./1.5e+1-real(t551.*t686.*t701.*(t93.*t491.*(t138+conj(t215).*conj(t251))-t94.*conj(t341).*conj(t371)).*(t31-t611))./1.5e+1-real(t552.*t687.*t702.*(t93.*t492.*(t138+conj(t217).*conj(t252))-t94.*conj(t342).*conj(t372)).*(t32-t612))./1.5e+1-real(t553.*t688.*t703.*(t93.*t493.*(t138+conj(t219).*conj(t253))-t94.*conj(t343).*conj(t373)).*(t33-t613))./1.5e+1-real(t554.*t689.*t704.*(t93.*t494.*(t138+conj(t221).*conj(t254))-t94.*conj(t344).*conj(t374)).*(t34-t614))./1.5e+1-real(t555.*t690.*t705.*(t93.*t495.*(t138+conj(t223).*conj(t255))-t94.*conj(t345).*conj(t375)).*(t35-t615))./1.5e+1-real(t556.*t691.*t706.*(t93.*t496.*(t138+conj(t225).*conj(t256))-t94.*conj(t346).*conj(t376)).*(t36-t616))./1.5e+1-real(t557.*t692.*t707.*(t93.*t497.*(t138+conj(t227).*conj(t257))-t94.*conj(t347).*conj(t377)).*(t37-t617))./1.5e+1-real(t558.*t693.*t708.*(t93.*t498.*(t138+conj(t228).*conj(t258))-t94.*conj(t348).*conj(t378)).*(t38-t618))./1.5e+1+real(-t514.*t649.*t664.*t739.*(t69.*t454.*(t128+t201.*t244)-t70.*t334.*t364))./1.5e+1+real(-t515.*t650.*t665.*t740.*(t69.*t455.*(t128+t203.*t245)-t70.*t335.*t365))./1.5e+1+real(-t516.*t651.*t666.*t741.*(t69.*t456.*(t128+t205.*t246)-t70.*t336.*t366))./1.5e+1+real(-t517.*t652.*t667.*t742.*(t69.*t457.*(t128+t207.*t247)-t70.*t337.*t367))./1.5e+1+real(-t518.*t653.*t668.*t743.*(t69.*t458.*(t128+t209.*t248)-t70.*t338.*t368))./1.5e+1+real(-t519.*t654.*t669.*t744.*(t69.*t459.*(t128+t211.*t249)-t70.*t339.*t369))./1.5e+1+real(-t520.*t655.*t670.*t745.*(t69.*t460.*(t128+t213.*t250)-t70.*t340.*t370))./1.5e+1+real(-t521.*t656.*t671.*t746.*(t69.*t461.*(t128+t215.*t251)-t70.*t341.*t371))./1.5e+1+real(-t522.*t657.*t672.*t747.*(t69.*t462.*(t128+t217.*t252)-t70.*t342.*t372))./1.5e+1+real(-t523.*t658.*t673.*t748.*(t69.*t463.*(t128+t219.*t253)-t70.*t343.*t373))./1.5e+1+real(-t524.*t659.*t674.*t749.*(t69.*t464.*(t128+t221.*t254)-t70.*t344.*t374))./1.5e+1+real(-t525.*t660.*t675.*t750.*(t69.*t465.*(t128+t223.*t255)-t70.*t345.*t375))./1.5e+1+real(-t526.*t661.*t676.*t751.*(t69.*t466.*(t128+t225.*t256)-t70.*t346.*t376))./1.5e+1+real(-t527.*t662.*t677.*t752.*(t69.*t467.*(t128+t227.*t257)-t70.*t347.*t377))./1.5e+1+real(-t528.*t663.*t678.*t753.*(t69.*t468.*(t128+t228.*t258)-t70.*t348.*t378))./1.5e+1;real(t694.*(t96.*t619.*conj(t574)-t95.*t679.*(conj(t379).*conj(t409)+conj(t394).*conj(t424))).*(t24-t604))./1.5e+1+real(t695.*(t96.*t620.*conj(t575)-t95.*t680.*(conj(t380).*conj(t410)+conj(t395).*conj(t425))).*(t25-t605))./1.5e+1+real(t696.*(t96.*t621.*conj(t576)-t95.*t681.*(conj(t381).*conj(t411)+conj(t396).*conj(t426))).*(t26-t606))./1.5e+1+real(t697.*(t96.*t622.*conj(t577)-t95.*t682.*(conj(t382).*conj(t412)+conj(t397).*conj(t427))).*(t27-t607))./1.5e+1+real(t698.*(t96.*t623.*conj(t578)-t95.*t683.*(conj(t383).*conj(t413)+conj(t398).*conj(t428))).*(t28-t608))./1.5e+1+real(t699.*(t96.*t624.*conj(t579)-t95.*t684.*(conj(t384).*conj(t414)+conj(t399).*conj(t429))).*(t29-t609))./1.5e+1+real(t700.*(t96.*t625.*conj(t580)-t95.*t685.*(conj(t385).*conj(t415)+conj(t400).*conj(t430))).*(t30-t610))./1.5e+1+real(t701.*(t96.*t626.*conj(t581)-t95.*t686.*(conj(t386).*conj(t416)+conj(t401).*conj(t431))).*(t31-t611))./1.5e+1+real(t702.*(t96.*t627.*conj(t582)-t95.*t687.*(conj(t387).*conj(t417)+conj(t402).*conj(t432))).*(t32-t612))./1.5e+1+real(t703.*(t96.*t628.*conj(t583)-t95.*t688.*(conj(t388).*conj(t418)+conj(t403).*conj(t433))).*(t33-t613))./1.5e+1+real(t704.*(t96.*t629.*conj(t584)-t95.*t689.*(conj(t389).*conj(t419)+conj(t404).*conj(t434))).*(t34-t614))./1.5e+1+real(t705.*(t96.*t630.*conj(t585)-t95.*t690.*(conj(t390).*conj(t420)+conj(t405).*conj(t435))).*(t35-t615))./1.5e+1+real(t706.*(t96.*t631.*conj(t586)-t95.*t691.*(conj(t391).*conj(t421)+conj(t406).*conj(t436))).*(t36-t616))./1.5e+1+real(t707.*(t96.*t632.*conj(t587)-t95.*t692.*(conj(t392).*conj(t422)+conj(t407).*conj(t437))).*(t37-t617))./1.5e+1+real(t708.*(t96.*t633.*conj(t588)-t95.*t693.*(conj(t393).*conj(t423)+conj(t408).*conj(t438))).*(t38-t618))./1.5e+1+real(-t649.*t739.*(t72.*t664.*(t379.*t409+t394.*t424)-t73.*t574.*t589))./1.5e+1+real(-t650.*t740.*(t72.*t665.*(t380.*t410+t395.*t425)-t73.*t575.*t590))./1.5e+1+real(-t651.*t741.*(t72.*t666.*(t381.*t411+t396.*t426)-t73.*t576.*t591))./1.5e+1+real(-t652.*t742.*(t72.*t667.*(t382.*t412+t397.*t427)-t73.*t577.*t592))./1.5e+1+real(-t653.*t743.*(t72.*t668.*(t383.*t413+t398.*t428)-t73.*t578.*t593))./1.5e+1+real(-t654.*t744.*(t72.*t669.*(t384.*t414+t399.*t429)-t73.*t579.*t594))./1.5e+1+real(-t655.*t745.*(t72.*t670.*(t385.*t415+t400.*t430)-t73.*t580.*t595))./1.5e+1+real(-t656.*t746.*(t72.*t671.*(t386.*t416+t401.*t431)-t73.*t581.*t596))./1.5e+1+real(-t657.*t747.*(t72.*t672.*(t387.*t417+t402.*t432)-t73.*t582.*t597))./1.5e+1+real(-t658.*t748.*(t72.*t673.*(t388.*t418+t403.*t433)-t73.*t583.*t598))./1.5e+1+real(-t659.*t749.*(t72.*t674.*(t389.*t419+t404.*t434)-t73.*t584.*t599))./1.5e+1+real(-t660.*t750.*(t72.*t675.*(t390.*t420+t405.*t435)-t73.*t585.*t600))./1.5e+1+real(-t661.*t751.*(t72.*t676.*(t391.*t421+t406.*t436)-t73.*t586.*t601))./1.5e+1+real(-t662.*t752.*(t72.*t677.*(t392.*t422+t407.*t437)-t73.*t587.*t602))./1.5e+1+real(-t663.*t753.*(t72.*t678.*(t393.*t423+t408.*t438)-t73.*t588.*t603))./1.5e+1];
