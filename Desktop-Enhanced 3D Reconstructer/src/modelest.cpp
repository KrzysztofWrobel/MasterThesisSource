///*M///////////////////////////////////////////////////////////////////////////////////////
////
////  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
////
////  By downloading, copying, installing or using the software you agree to this license.
////  If you do not agree to this license, do not download, install,
////  copy or use the software.
////
////
////                        Intel License Agreement
////                For Open Source Computer Vision Library
////
//// Copyright (C) 2000, Intel Corporation, all rights reserved.
//// Third party copyrights are property of their respective owners.
////
//// Redistribution and use in source and binary forms, with or without modification,
//// are permitted provided that the following conditions are met:
////
////   * Redistribution's of source code must retain the above copyright notice,
////     this list of conditions and the following disclaimer.
////
////   * Redistribution's in binary form must reproduce the above copyright notice,
////     this list of conditions and the following disclaimer in the documentation
////     and/or other materials provided with the distribution.
////
////   * The name of Intel Corporation may not be used to endorse or promote products
////     derived from this software without specific prior written permission.
////
//// This software is provided by the copyright holders and contributors "as is" and
//// any express or implied warranties, including, but not limited to, the implied
//// warranties of merchantability and fitness for a particular purpose are disclaimed.
//// In no event shall the Intel Corporation or contributors be liable for any direct,
//// indirect, incidental, special, exemplary, or consequential damages
//// (including, but not limited to, procurement of substitute goods or services;
//// loss of use, data, or profits; or business interruption) however caused
//// and on any theory of liability, whether in contract, strict liability,
//// or tort (including negligence or otherwise) arising in any way out of
//// the use of this software, even if advised of the possibility of such damage.
////
////M*/
//
#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"

#include "modelest.h"

#include <algorithm>
#include <iterator>
#include <limits>
//
using namespace std;
using namespace cv;


bool checkPartialSubsets = false;

int cvRANSACUpdateNumItersy(double p, double ep,
        int model_points, int max_iters) {
    if (model_points <= 0)
        CV_Error(CV_StsOutOfRange, "the number of model points should be positive");

    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    double num = MAX(1. - p, DBL_MIN);
    double denom = 1. - pow(1. - ep, model_points);
    if (denom < DBL_MIN)
        return 0;

    num = log(num);
    denom = log(denom);

    return denom >= 0 || -num >= max_iters * (-denom) ?
            max_iters : cvRound(num / denom);
}


int findInliersy(const vector<Point2d> &m1, const vector<Point2d> &m2,
        Mat F, float* err,
        uchar* mask, double threshold) {
    int i = 0;
    int pointNumber = m1.size();
    int gCount = 0;

    computeReprojErrory(m1, m2, F, err);
    threshold *= threshold;
    for (i = 0; i < pointNumber; i++) {
        gCount += mask[i] = err[i] <= threshold;
    }
    return gCount;
}


void computeReprojErrory(const vector<Point2d> &m1, const vector<Point2d> &m2,
        Mat F, float* err) {
    int i = 0;
    int size = m1.size();

    for (i = 0; i < size; i++) {
        double a, b, c, d1, d2, s1, s2;

        a = F.at<double>(0) * m1[i].x + F.at<double>(1) * m1[i].y + F.at<double>(2);
        b = F.at<double>(3) * m1[i].x + F.at<double>(4) * m1[i].y + F.at<double>(5);
        c = F.at<double>(6) * m1[i].x + F.at<double>(7) * m1[i].y + F.at<double>(8);

        s2 = 1. / (a * a + b * b);
        d2 = m2[i].x * a + m2[i].y * b + c;

        a = F.at<double>(0) * m2[i].x + F.at<double>(3) * m2[i].y + F.at<double>(6);
        b = F.at<double>(1) * m2[i].x + F.at<double>(4) * m2[i].y + F.at<double>(7);
        c = F.at<double>(2) * m2[i].x + F.at<double>(5) * m2[i].y + F.at<double>(8);

        s1 = 1. / (a * a + b * b);
        d1 = m1[i].x * a + m1[i].y * b + c;

        err[i] = (float) std::max(d1 * d1 * s1, d2 * d2 * s2);
    }
}

bool getSubsety(const vector<Point2d> &m1, const vector<Point2d> &m2,
        vector<Point2d> &ms1, vector<Point2d> &ms2, int maxAttempts, int modelPoints) {
    CvRNG rng(cv::getTickCount());
    cv::AutoBuffer<int> _idx(modelPoints);
    int *idx = _idx;
    int i = 0, j, k, idx_i, iters = 0;
    int count = m1.size();

    ms1.reserve(modelPoints);
    ms2.reserve(modelPoints);

    vector<Point2d>::const_iterator m1ptr = m1.begin(), m2ptr = m2.begin();
    vector<Point2d>::iterator ms1ptr = ms1.begin(), ms2ptr = ms2.begin();

    for (; iters < maxAttempts; iters++) {
        ms1.clear();
        ms2.clear();
        for (i = 0; i < modelPoints && iters < maxAttempts;) {
            idx[i] = idx_i = cvRandInt(&rng) % count;
            for (j = 0; j < i; j++)
                if (idx_i == idx[j])
                    break;
            if (j < i)
                continue;
            ms1ptr[i] = m1ptr[idx_i];
            ms2ptr[i] = m2ptr[idx_i];
            ms1.push_back((m1ptr[idx_i]));
            ms2.push_back((m2ptr[idx_i]));
//            if( checkPartialSubsets && (!checkSubset( ms1, i+1 ) || !checkSubset( ms2, i+1 )))
//            {
//                iters++;
//                continue;
//            }
            i++;
        }
        if (i == modelPoints &&
                (!checkSubsety(ms1, i) || !checkSubsety(ms2, i)))
            continue;
        break;
    }

    return i == modelPoints && iters < maxAttempts;
}


bool checkSubsety(const vector<Point2d> &ptr, int count) {
    int j, k, i, i0, i1;

    if (checkPartialSubsets)
        i0 = i1 = count - 1;
    else
        i0 = 0, i1 = count - 1;

    for (i = i0; i <= i1; i++) {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for (j = 0; j < i; j++) {
            double dx1 = ptr[j].x - ptr[i].x;
            double dy1 = ptr[j].y - ptr[i].y;
            for (k = 0; k < j; k++) {
                double dx2 = ptr[k].x - ptr[i].x;
                double dy2 = ptr[k].y - ptr[i].y;
                if (fabs(dx2 * dy1 - dy2 * dx1) <= FLT_EPSILON * (fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if (k < j)
                break;
        }
        if (j < i)
            break;
    }

    return i >= i1;
}

//bool runRANSAC( const CvMat* m1, const CvMat* m2, CvMat* model,
//        CvMat* mask0, double reprojThreshold,
//        double confidence, int maxIters )
//{
//    bool result = false;
//    cv::Ptr<CvMat> mask = cvCloneMat(mask0);
//    cv::Ptr<CvMat> models, err, tmask;
//    cv::Ptr<CvMat> ms1, ms2;
//
//    int iter, niters = maxIters;
//    int count = m1->rows*m1->cols, maxGoodCount = 0;
//    CV_Assert( CV_ARE_SIZES_EQ(m1, m2) && CV_ARE_SIZES_EQ(m1, mask) );
//
//    if( count < modelPoints )
//        return false;
//
//    models = cvCreateMat( modelSize.height*maxBasicSolutions, modelSize.width, CV_64FC1 );
//    err = cvCreateMat( 1, count, CV_32FC1 );
//    tmask = cvCreateMat( 1, count, CV_8UC1 );
//
//    if( count > modelPoints )
//    {
//        ms1 = cvCreateMat( 1, modelPoints, m1->type );
//        ms2 = cvCreateMat( 1, modelPoints, m2->type );
//    }
//    else
//    {
//        niters = 1;
//        ms1 = cvCloneMat(m1);
//        ms2 = cvCloneMat(m2);
//    }
//
//    for( iter = 0; iter < niters; iter++ )
//    {
//        int i, goodCount, nmodels;
//        if( count > modelPoints )
//        {
//            bool found = getSubset( m1, m2, ms1, ms2, 300 );
//            if( !found )
//            {
//                if( iter == 0 )
//                    return false;
//                break;
//            }
//        }
//
//        nmodels = runKernel( ms1, ms2, models );
//        if( nmodels <= 0 )
//            continue;
//        for( i = 0; i < nmodels; i++ )
//        {
//            CvMat model_i;
//            cvGetRows( models, &model_i, i*modelSize.height, (i+1)*modelSize.height );
//            goodCount = findInliers( m1, m2, &model_i, err, tmask, reprojThreshold );
//
//            if( goodCount > MAX(maxGoodCount, modelPoints-1) )
//            {
//                std::swap(tmask, mask);
//                cvCopy( &model_i, model );
//                maxGoodCount = goodCount;
//                niters = cvRANSACUpdateNumIters( confidence,
//                        (double)(count - goodCount)/count, modelPoints, niters );
//            }
//        }
//    }
//
//    if( maxGoodCount > 0 )
//    {
//        if( mask != mask0 )
//            cvCopy( mask, mask0 );
//        result = true;
//    }
//
//    return result;
//}

