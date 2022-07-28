%{
========================================================================
   Independent Feature Similarity (IFS) Version 1.0
   Copyright(c) 2014  Hua-wen Chang
   All Rights Reserved.
   changhuawen@gmail.com or
   changhuawen@126.com
------------------------------------------------------------------------
 Permission to use, copy, or modify this software and its documentation
 for educational and research purposes only and without fee is here
 granted, provided that this copyright notice and the original authors'
 names appear on all copies and supporting documentation. This program
 shall not be used, rewritten, or adapted as the basis of a commercial
 software or hardware product without first obtaining permission of the
 authors. The authors make no representations about the suitability of
 this software for any purpose. It is provided "as is" without express
 or implied warranty.
------------------------------------------------------------------------
 Please refer to the following paper

 Hua-wen Chang, Qiu-wen Zhang, Qing-gang Wu, and Yong Gan,"Perceptual image
 quality assessment by independent feature detector", Neurocomputing,
 vol. 151, pp. 1142-1152, March 2015
------------------------------------------------------------------------
INPUT variables:
 Ir:       reference color image
 Id:       distorted color image
 iW:       independent feature detector (provided in 'iW.mat')

 Example:
 load('iW.mat');        % load feature detector, iW
 score = IFS(refImage, disImage, iW);
 =================================================================
%}

function ifsScore = IFS(Ir,Id,iW)
patchSize = 8;
%%%% Parameters
Cm = 0.001;
C = 0.12;
Tm = 0.8;
Tx = 7;
patchDim = (patchSize^2)*3;
startPosition = 1;
overlap = 0;

sizeY = size(Ir,1);
sizeX = size(Ir,2);

%%%     DIVIDING EACH IMAGE INTO BLOCKS    %%%
gridY = startPosition : patchSize-overlap : sizeY-patchSize; %
gridX = startPosition : patchSize-overlap : sizeX-patchSize; %
Y = length(gridY);  X = length(gridX);
Xr = zeros(patchDim, Y*X);
Xd = zeros(patchDim, Y*X);
ij = 0;
for i = gridY;
    for j = gridX
        ij = ij+1;
        Xr(:,ij) = reshape( Ir(i:i+patchSize-1, j:j+patchSize-1, 1:3), [patchDim 1] );
        Xd(:,ij) = reshape( Id(i:i+patchSize-1, j:j+patchSize-1, 1:3), [patchDim 1] );
    end
end
Xr = double(Xr);        Xd = double(Xd);
MXr = mean(Xr);         MXd = mean(Xd);
Xr_noMean = removeMean(Xr); Xd_noMean = removeMean(Xd);

%%%     THRESHOLD FOR BLOCKS     %%%
Xe = mean(abs(Xr_noMean-Xd_noMean));
med_Xe = median(Xe);
max_Xe = max(Xe);

Tsize = sizeX*sizeY/(512*512);
T = Tsize*Tx;

if med_Xe < T
    TH = med_Xe;
else
    TH = (max_Xe+4*med_Xe)/(5);
end

Xd_noMean = Xd_noMean(:,Xe>=TH);
Xr_noMean = Xr_noMean(:,Xe>=TH);

%%%     FEATURE EXTRACTION     %%%
Sdis = iW*Xd_noMean;
Sref = iW*Xr_noMean;

%%%    INDEPENDENT FEATURES    %%%
ifs1 = Sdis.*Sref;
ifs2 = Sdis.^2+Sref.^2;
Qifs = mean2((2*ifs1+C)./(ifs2+C));

%%%      LUMINANCE VALUE       %%%
MXe = abs(MXr-MXd);
[MXe_sorted,MXe_index] = sort(MXe);
Front = floor(ij*Tm); End = floor(ij);
if Front < 1
    Front = 1;
end
MXr = MXr(MXe_index(Front:End));
MXd = MXd(MXe_index(Front:End));
meanXr = MXr-mean(MXr);
meanXd = MXd-mean(MXd);
Qlum = (sum(meanXr.*meanXd)+Cm) / (sqrt(sum(meanXr.^2)*sum(meanXd.^2))+Cm);

%%%    IFS QUALITY INDEX    %%%
ifsScore = sqrt(Qifs*Qlum);
return;


%%%%%%%  Remove Mean  %%%%%%
function Y = removeMean(X)
Y = X-ones(size(X,1),1)*mean(X);
return;

