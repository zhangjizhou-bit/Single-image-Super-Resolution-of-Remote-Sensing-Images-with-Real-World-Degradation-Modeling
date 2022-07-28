=======================================================================
 Perceptual Image Quality Assessment by Independent Feature Detector
=======================================================================

This package contains Matlab codes for the Independent Feature Similarity (IFS) quality index.
IFS is a new algorithm for evaluating perceptual quality of color images.

Plase use the citation provided below if it is useful to your research:

Hua-wen Chang, Qiu-wen Zhang, Qing-gang Wu, and Yong Gan,"Perceptual image 
quality assessment by independent feature detector", Neurocomputing, 
vol. 151, pp. 1142-1152, March 2015


------------------------- COPYRIGHT NOTICE --------------------------
Copyright (c) 2014, Hua-wen Chang
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright 
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the distribution

---------------------------------------------------------------------



---------------------------------------------------
For quality evaluation, you should load feature detector (iW.mat) before running 'IFS'.

EXAMPLE:
load('iW.mat');  % load the feature detector which is in 'iW.mat'
score = IFS(refImg, disImg, iW);  % refImg and disImg respectively denote the reference and distorted color images

The quality scores are between 0 and 1, where 1 represents the same quality as the reference image.
---------------------------------------------------



---------------------------------------------------
Moreover, 'IFS_on_6DBs.mat' provides all the results (quality scores) of IFS on six databases, 
including CSIQ, IVC, LIVE, TID2008, TID2013 and Toyama MICT.
There are six data groups in this file, each group corresponds to the results on a database.
*_ifs is the result of IFS; *_mos is the subjective ratings of each database.

LIVE: 		[ live_ifs  	live_mos ]
CSIQ: 		[ csiq_ifs  	csiq_mos ]
IVC: 		[ ivc_ifs   	ivc_mos  ]
TID2008:	[ tid2008_ifs   tid2008_mos ]
TID2013:	[ tid2013_ifs   tid2013_mos ]
Toyama-MICT:	[ toy_ifs   	toy_mos ]
---------------------------------------------------


Please send any questions about this metric to
changhuawen@gmail.com
or
changhuawen@126.com

Hua-wen Chang
Feb. 6th, 2015.