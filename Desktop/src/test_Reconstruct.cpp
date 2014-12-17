/* Example 10-1. Pyramid Lucas-Kanade optical flow code

 ************************************************** */
#include <iostream>
#include <fstream>
#include <cassert>
#include <exception>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PCL
#ifdef PCL

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#endif

#include "Utils64.h"
#include "IOUtils.h"
#include "TestUtils.h"
#include "FeatureMatching.h"
#include "MultiView.h"
#include "modelest.h"

#include <stdio.h>
#include <time.h>
#include <src/five-point-nister/five-point.hpp>
#include <cvsba/cvsba.h>

#define DEBUG

using namespace std;
using namespace cv;

enum INIT_METHOD {
    FUNDAMENTAL = 0,
    FUNDAMENTAL_ENHENCED,
    ESSENTIAL,
    ESSENTIAL_ENHANCED,
    TRANSLATION_ESTIM,
    NONE_INIT
};

enum POSE_ESTIM_METHOD {
    NORMAL = 0,
    ROTATION_ENHENCED,
    ROT_TRANS_ENHANCED,
    NONE_POSE
};

struct WrapAff {
    const double *F;

    WrapAff(const Mat &aff) : F(aff.ptr<double>()) {
    }

    Point3f operator()(const Point3f &p) {
        return Point3d(p.x * F[0] + p.y * F[1] + p.z * F[2] + F[3],
                p.x * F[4] + p.y * F[5] + p.z * F[6] + F[7],
                p.x * F[8] + p.y * F[9] + p.z * F[10] + F[11]);
    }
};

//TODO when bundleAdjustments
int main(int argc, char **argv) {

    //TODO safe parsing input while and modulo
    cout << "Please enter Sift features number: ";
    int SIFT_FEATURES = 200;
    cin >> SIFT_FEATURES;

    cout << "Please choose Init Reconstruct Method:\n"
            "    0 - Fundamental,\n"
            "    1 - FUNDAMENTAL_ENHENCED,\n"
            "    2 - ESSENTIAL,\n"
            "    3 - ESSENTIAL_ENHANCED,\n"
            "    4 - TRANSLATION_ESTIM,\n"
            "    5 - NONE_INIT\n";

    int method = FUNDAMENTAL;
    std::cin >> method;
    INIT_METHOD init_method = static_cast<INIT_METHOD>( method );

    cout << "Please choose Pose Estimation Method:\n"
            "    0 - NORMAL,\n"
            "    1 - NORMAL_ENHENCED,\n"
            "    2 - TRANSLATION_POSE,\n"
            "    3 - NONE_POSE\n";
    int poseMethod = NORMAL;
    std::cin >> poseMethod;
    POSE_ESTIM_METHOD estim_method = static_cast<POSE_ESTIM_METHOD>( poseMethod );

    //TODO ask for outliers
    bool removeOutliers = true;
    bool removeOutliersPose = true;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> cameraKMatrix;
    std::vector<cv::Mat> distCoeffsMatrix;
    std::vector<std::vector<int> > visiblity;
    std::vector<cv::Mat> R_true, R_init, R_opt;
    std::vector<cv::Mat> T_true, T_init, T_opt;
    std::vector<cv::Mat> cameraDistCoeffs;

    Mat K, distCoeffs;
    std::vector<ImageDesc> imageDescriptions;
    char const *sensorDataFilename = "sensor.txt";
    char const *cameraIntersincFilename = "out_camera_data.yml";

    parseSensorFile(sensorDataFilename, imageDescriptions);
    parseCameraInstersincParameters(cameraIntersincFilename, K, distCoeffs);
    Mat KInv = K.inv();


    Mat prev_frame, next_frame;
    vector<KeyPoint> prev_keypoints, next_keypoints;
    Mat prev_descriptors, next_descriptors;

    std::vector<uchar> status;
    std::vector<double> err;

    vector<vector<DMatch> > good_matchesVec;
    vector<DMatch> good_matches;
    vector<Point2d> prev_points_raw, next_points_raw;

    int imageCount = imageDescriptions.size();
    for (int i = 0; i < imageCount; i++) {
        images.push_back(imread(imageDescriptions[i].path, CV_LOAD_IMAGE_COLOR));   // Read the file
    }

    int startIdx = 0;
    int endIdx = imageCount - 1;
    int prevIdx = 0;
    int nextIdx = 1;

    cvtColor(images[prevIdx], prev_frame, CV_RGB2GRAY);
    GaussianBlur(prev_frame, prev_frame, Size(3, 3), 1.5, 1.5);
    getSiftKeypointsAndDescriptors(prev_frame, prev_keypoints, prev_descriptors, SIFT_FEATURES);

    cvtColor(images[nextIdx], next_frame, CV_RGB2GRAY);
    GaussianBlur(next_frame, next_frame, Size(3, 3), 1.5, 1.5);
    getSiftKeypointsAndDescriptors(next_frame, next_keypoints, next_descriptors, SIFT_FEATURES);

    getCorrespondingPoints(prev_keypoints, next_keypoints, prev_descriptors, next_descriptors, good_matches, prev_points_raw, next_points_raw);

    Mat rotDiffGlobal;
    double pitchDiff, azimuthDiff, rollDiff;
    getRelativeMatrixAndAngles(imageDescriptions, prevIdx, nextIdx, rotDiffGlobal, pitchDiff, azimuthDiff, rollDiff);

    Mat R, T;
    int fundEssenEstimationMethod = FM_RANSAC; //FM_RANSAC FM_LMEDS
    std::chrono::high_resolution_clock::time_point tInitStart = std::chrono::high_resolution_clock::now();

    Mat R1, R2;
    Mat t1, t2;
    double minVal, maxVal;
    cout << "initMethod" << endl;
    switch (init_method) {
        case FUNDAMENTAL: {
            cv::minMaxIdx(prev_points_raw, &minVal, &maxVal);
            std::chrono::high_resolution_clock::time_point ti1 = std::chrono::high_resolution_clock::now();
//            findFundamentalStandard(prev_points_raw, next_points_raw, FStandard, goodStatuses);
            Mat F = findFundamentalMat(prev_points_raw, next_points_raw, fundEssenEstimationMethod, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
            F.convertTo(F, CV_64FC1);

            Mat E = K.t() * F * K;
            decomposeEssentialMat(E, R1, R2, t1);
            t2 = -t1;

            chooseProperRAndTFromTriangulation(prev_points_raw, next_points_raw, K, distCoeffs, R1, R2, t1, t2, R, T);
            break;
        };
        case FUNDAMENTAL_ENHENCED: {
            Mat dR;
            Mat TProp;
            Mat TdRExp = findFundamentalEnhanced(prev_points_raw, next_points_raw, K, distCoeffs, fundEssenEstimationMethod, rotDiffGlobal, dR, TProp, status);
            R1 = dR * rotDiffGlobal;
            t1 = TProp;
            t2 = -TProp;
            //TODO decide T from triangulation ???
            chooseProperRAndTFromTriangulation(prev_points_raw, next_points_raw, K, distCoeffs, R1, R1, t1, t2, R, T);
            break;

        };
        case ESSENTIAL: {
            Mat points1ess5(prev_points_raw.size(), 3, CV_64FC1), points2ess5(next_points_raw.size(), 3, CV_64FC1);
            for (int a = 0; a < prev_points_raw.size(); a++) {
                points1ess5.at<double>(a, 0) = prev_points_raw[a].x;
                points1ess5.at<double>(a, 1) = prev_points_raw[a].y;
                points1ess5.at<double>(a, 2) = 1;
                points2ess5.at<double>(a, 0) = next_points_raw[a].x;
                points2ess5.at<double>(a, 1) = next_points_raw[a].y;
                points2ess5.at<double>(a, 2) = 1;
            }
            CV_Assert(points1ess5.type() == CV_64FC1);
            cv::minMaxIdx(prev_points_raw, &minVal, &maxVal);
            Mat EssentialReal = findEssentialMat(points1ess5, points2ess5, K.at<double>(0), Point2d(K.at<double>(0, 2), K.at<double>(1, 2)), fundEssenEstimationMethod, 0.99, 0.006 * maxVal, status); //threshold from [Snavely07 4.1]

            decomposeEssentialMat(EssentialReal, R1, R2, t1);
            t2 = -t1;

            chooseProperRAndTFromTriangulation(prev_points_raw, next_points_raw, K, distCoeffs, R1, R2, t1, t2, R, T);
            break;
        };
        case ESSENTIAL_ENHANCED: {
            Mat points1ess5en(prev_points_raw.size(), 3, CV_64FC1), points2ess5en(next_points_raw.size(), 3, CV_64FC1);
            for (int a = 0; a < prev_points_raw.size(); a++) {
                points1ess5en.at<double>(a, 0) = prev_points_raw[a].x;
                points1ess5en.at<double>(a, 1) = prev_points_raw[a].y;
                points1ess5en.at<double>(a, 2) = 1;
                points2ess5en.at<double>(a, 0) = next_points_raw[a].x;
                points2ess5en.at<double>(a, 1) = next_points_raw[a].y;
                points2ess5en.at<double>(a, 2) = 1;
            }
            Mat EssentialEnhanced = findEssentialMatEnhanced(points1ess5en, points2ess5en, K, rotDiffGlobal, fundEssenEstimationMethod, status);

            decomposeEssentialMat(EssentialEnhanced, R1, R2, t1);
            R1 = R1 * rotDiffGlobal;
            R2 = R2 * rotDiffGlobal;
            t2 = -t1;
            chooseProperRAndTFromTriangulation(prev_points_raw, next_points_raw, K, distCoeffs, R1, R2, t1, t2, R, T);
            break;
        };
        case TRANSLATION_ESTIM: {
            //Correction check
//            Mat dR;
//            Mat TProp;
//            Mat TdRExp = findFundamentalEnhanced(prev_points_raw, next_points_raw, K, distCoeffs, fundEssenEstimationMethod, rotDiffGlobal, dR, TProp, status);
//            rotDiffGlobal = dR * rotDiffGlobal;

            Mat FEnhanced;
            Mat tEnhanced;
            uchar *goodStatuses;
            getTranslationWithKnownRotation(prev_points_raw, next_points_raw, K, rotDiffGlobal, FEnhanced, tEnhanced, goodStatuses);
            status = std::vector<uchar>(goodStatuses, goodStatuses + prev_points_raw.size());

            R1 = rotDiffGlobal;
            t1 = tEnhanced;
            t2 = -tEnhanced;
            chooseProperRAndTFromTriangulation(prev_points_raw, next_points_raw, K, distCoeffs, R1, R1, t1, t2, R, T);
            break;
        };
        case NONE_INIT: {
            R = rotDiffGlobal;
            T = Mat(3, 1, CV_64FC1);
            T.at<double>(0) = imageDescriptions[nextIdx].globalPosX;
            T.at<double>(1) = imageDescriptions[nextIdx].globalPosY;
            T.at<double>(2) = imageDescriptions[nextIdx].globalPosZ;
            break;
        };
    }
    std::chrono::high_resolution_clock::time_point tInitEnd = std::chrono::high_resolution_clock::now();
    double durationInit = std::chrono::duration_cast<std::chrono::microseconds>(tInitEnd - tInitStart).count();
    cout << "Duration init reconstruct: " << durationInit << endl;

    if (removeOutliers) {
        cout << "Keeping " << countNonZero(status) << " / " << status.size() << endl;
        for (int i = status.size() - 1; i >= 0; i--) {
            if (!status[i]) {
                prev_points_raw.erase(prev_points_raw.begin() + i);
                next_points_raw.erase(next_points_raw.begin() + i);
                good_matches.erase(good_matches.begin() + i);
            }
        }
    }


    Mat img_matches;
    drawMatches(prev_frame, prev_keypoints, next_frame, next_keypoints, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good MatchesInit", img_matches);

    Matx34d P1, P2;
    P1 = Matx34d::eye();
    P2 = Matx34d(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));

    vector<Point2d> points1StUD, points2StUD;
    cout << "Undistort!" << endl;
    undistortPoints(prev_points_raw, points1StUD, K, distCoeffs);
    undistortPoints(next_points_raw, points2StUD, K, distCoeffs);

    Mat reconstructCloud = Mat::zeros(4, 0, CV_64FC1);
    cout << "Traingulate!" << endl;
    vector<Point3d> reconstructCloudVec;
    triangulatePoints(P1, P2, points1StUD, points2StUD, reconstructCloud);
//    triangulatePoints(P1, P2, points1StUD, points2StUD, reconstructCloudVec);
    vector<Point3d> xp_p;
    Mat reshape = Mat::zeros(0, 4, CV_64FC1);
    MatExpr reconstructTransposed = reconstructCloud.t();
    convertPointsFromHomogeneous(reconstructTransposed, reconstructCloudVec);
    convertPointsFromHomogeneous(reconstructTransposed, xp_p);
    reshape.push_back(Mat(reconstructTransposed));
    cout << reshape << endl;
    vector<vector<int> >img_pt_matches;
    vector<vector<Point2d> >img_pt_matches_pts;
    
    for (int i = 0; i < reconstructCloudVec.size(); i++) {
        vector<int> img_pt_match(imageCount);
        img_pt_match[prevIdx] = good_matches[i].queryIdx;
        img_pt_match[nextIdx] = good_matches[i].trainIdx;
        img_pt_matches.push_back(img_pt_match);
        vector<Point2d> img_pt_match_pt(imageCount);
        img_pt_match_pt[prevIdx] = prev_keypoints[good_matches[i].queryIdx].pt;
        img_pt_match_pt[nextIdx] = next_keypoints[good_matches[i].trainIdx].pt;
        img_pt_matches_pts.push_back(img_pt_match_pt);
    }
    good_matchesVec.push_back(vector<DMatch>(good_matches));
    
    R_opt.push_back(Mat::eye(3,3, CV_64FC1));
    T_opt.push_back(Mat::zeros(3, 1, CV_64FC1));
    R_opt.push_back(R);
    T_opt.push_back(T);

    vector<Point3d> prev_xp_p = xp_p;
    prev_xp_p = vector<Point3d>(xp_p);
    prev_frame = next_frame.clone();
    prev_keypoints = vector<KeyPoint>(next_keypoints);
    prev_descriptors = next_descriptors.clone();

    for (int nextIdx = startIdx + 2; nextIdx <= endIdx; nextIdx++) {
        good_matches.clear();
        prev_points_raw.clear();
        next_points_raw.clear();
        int prevIdx = nextIdx - 1;

        Mat prev_rot = R.clone();
        Mat prev_trans = T.clone();
        Mat rvec;

        cvtColor(images[nextIdx], next_frame, CV_RGB2GRAY);
        GaussianBlur(next_frame, next_frame, Size(3, 3), 1.5, 1.5);
        getSiftKeypointsAndDescriptors(next_frame, next_keypoints, next_descriptors, SIFT_FEATURES);

        getCorrespondingPoints(prev_keypoints, next_keypoints, prev_descriptors, next_descriptors, good_matches, prev_points_raw, next_points_raw);

        Mat rotDiffGlobal;
        double pitchDiff, azimuthDiff, rollDiff;
        getRelativeMatrixAndAngles(imageDescriptions, prevIdx, nextIdx, rotDiffGlobal, pitchDiff, azimuthDiff, rollDiff);

        Mat img_matches;
        drawMatches(prev_frame, prev_keypoints, next_frame, next_keypoints, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        char matchbuff[100];
        sprintf(matchbuff, "Matches - %d : %d", prevIdx, nextIdx);
        std::string matchbuffAsStdStr = matchbuff;
        imshow(matchbuffAsStdStr, img_matches);

        //TODO eliminate outliers

        cout << "begin:" << endl;
        vector<Point3d> temp3D;
        vector<Point2d> temp2D;
        vector<int> inliers;
        vector<int> newPointIndexes;
        for (int i = 0; i < good_matches.size(); i++) {
            bool found = false;
            for(int pt3d = 0; pt3d < reconstructCloudVec.size(); pt3d++){
                if(img_pt_matches[pt3d][prevIdx] == good_matches[i].queryIdx){
                    img_pt_matches[pt3d][nextIdx] = good_matches[i].trainIdx;
                    img_pt_matches_pts[pt3d][nextIdx] = next_keypoints[good_matches[i].trainIdx].pt;
                    temp3D.push_back(reconstructCloudVec[pt3d]);
                    Point2f point2f = next_keypoints[good_matches[i].trainIdx].pt;
                    temp2D.push_back(Point2d(point2f.x, point2f.y));
                    cout << "prevCloudKeypoints[i]: " << good_matches[i].queryIdx << " good_matches[j].trainIdx: " << good_matches[i].trainIdx << endl;
                    found = true;
                    break;
                }
            }
//            bool found = false;
//            for (int i = 0; i < good_matchesVec[prevIdx-1].size(); i++) {
//                if (good_matchesVec[prevIdx-1][i].trainIdx == good_matches[j].queryIdx) {
//
//                    found = true;
//                    break;
//                }
//            }
//            if (!found) {
//                newPointIndexes.push_back(j);
//            }
        }
        Mat RInit = prev_rot * rotDiffGlobal;
        Mat Tnext = Mat(3, 1, CV_64FC1);
        Tnext.at<double>(0) = imageDescriptions[nextIdx].globalPosX;
        Tnext.at<double>(1) = imageDescriptions[nextIdx].globalPosY;
        Tnext.at<double>(2) = imageDescriptions[nextIdx].globalPosZ;
        switch(poseMethod){
            case NORMAL:
            {
                FindPoseEstimation(rvec, T, R, K, distCoeffs, temp3D, temp2D, inliers);
                break;
            };
            case ROTATION_ENHENCED:
            {
                Mat TInit = Mat::zeros(3, 1, CV_64FC1);
                FindPoseEstimationEnhanced(rvec, T, R, RInit, TInit, K, distCoeffs, temp3D, temp2D, inliers);
                break;
            };
            case ROT_TRANS_ENHANCED:
            {
                Mat TInit = prev_trans + Tnext;
                FindPoseEstimationEnhanced(rvec, T, R, RInit, TInit, K, distCoeffs, temp3D, temp2D, inliers);
                break;
            };
            case NONE_POSE:
            {
                R = RInit;
                T = prev_trans + Tnext;
                break;
            };
        }

        R_opt.push_back(R);
        T_opt.push_back(T);

        P1 = Matx34d(prev_rot.at<double>(0, 0), prev_rot.at<double>(0, 1), prev_rot.at<double>(0, 2), prev_trans.at<double>(0),
                prev_rot.at<double>(1, 0), prev_rot.at<double>(1, 1), prev_rot.at<double>(1, 2), prev_trans.at<double>(1),
                prev_rot.at<double>(2, 0), prev_rot.at<double>(2, 1), prev_rot.at<double>(2, 2), prev_trans.at<double>(2));
        P2 = Matx34d(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), T.at<double>(0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), T.at<double>(1),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), T.at<double>(2));

        undistortPoints(prev_points_raw, points1StUD, K, distCoeffs);
        undistortPoints(next_points_raw, points2StUD, K, distCoeffs);

        Mat XP = Mat::zeros(0, 4, CV_64FC1);
        cout << "Traingulate!" << endl;


        triangulatePoints(P1, P2, points1StUD, points2StUD, XP);
        vector<Point3d> xp_p;
        Mat xpreshape = XP.t();
        convertPointsFromHomogeneous(xpreshape, xp_p);

        cout << R << endl;
        cout << T << endl;

        //Filter estimated points from new and outliers, we should think about updating xp_p points also probably, because it can correlete in between outliers pairs
        if(removeOutliersPose) {
            for (int j = 0; j < inliers.size(); j++) {
                    for (int i = 0; i < xp_p.size(); i++) {
                        if(i == inliers[j]){
                            bool found = false;
                            for(int pt3d = 0; pt3d < reconstructCloudVec.size(); pt3d++){
                                if(img_pt_matches[pt3d][prevIdx] == good_matches[i].queryIdx){
                                    img_pt_matches[pt3d][nextIdx] = good_matches[i].trainIdx;
                                    found = true;
                                    break;
                                }
                            }
                            if(!found){
                                Mat elem = Mat(xpreshape.row(i));
                                reshape.push_back(elem);
                                reconstructCloudVec.push_back(xp_p[i]);
                                vector<int> img_pt_match(imageCount, 0);
                                img_pt_match[prevIdx] = good_matches[i].queryIdx;
                                img_pt_match[nextIdx] = good_matches[i].trainIdx;
                                img_pt_matches.push_back(img_pt_match);
                                vector<Point2d> img_pt_match_pt(imageCount);
                                img_pt_match_pt[prevIdx] = prev_keypoints[good_matches[i].queryIdx].pt;
                                img_pt_match_pt[nextIdx] = next_keypoints[good_matches[i].trainIdx].pt;
                                img_pt_matches_pts.push_back(img_pt_match_pt);
                            }
                        }
                    }
                }

        } else {
            for (int i = 0; i < xp_p.size(); i++) {
                bool found = false;
                for(int pt3d = 0; pt3d < reconstructCloudVec.size(); pt3d++){
                    if(img_pt_matches[pt3d][prevIdx] == good_matches[i].queryIdx){
                        img_pt_matches[pt3d][nextIdx] = good_matches[i].trainIdx;
                        found = true;
                        break;
                    }
                }
                if(!found){
                    Mat elem = Mat(xpreshape.row(i));
                    reshape.push_back(elem);
                    reconstructCloudVec.push_back(xp_p[i]);
                    vector<int> img_pt_match(imageCount, 0);
                    img_pt_match[prevIdx] = good_matches[i].queryIdx;
                    img_pt_match[nextIdx] = good_matches[i].trainIdx;
                    img_pt_matches.push_back(img_pt_match);
                    vector<Point2d> img_pt_match_pt(imageCount);
                    img_pt_match_pt[prevIdx] = prev_keypoints[good_matches[i].queryIdx].pt;
                    img_pt_match_pt[nextIdx] = next_keypoints[good_matches[i].trainIdx].pt;
                    img_pt_matches_pts.push_back(img_pt_match_pt);
                }
            }

        }
        prev_xp_p = vector<Point3d>(xp_p);
        prev_frame = next_frame.clone();
        prev_keypoints = vector<KeyPoint>(next_keypoints);
        prev_descriptors = next_descriptors.clone();
        cout << reshape.rows << endl;
//        reconstructCloud.push_back(Mat(XP.t()));

        char buff[100];
        sprintf(buff, "projectiveReconstructionStep%d.asc", nextIdx);
        std::string buffAsStdStr = buff;
        savePointList(buffAsStdStr, XP);
    }

    Mat reshapeHomo;
    convertPointsFromHomogeneous(reshape, reshapeHomo);
    cv::Size cameraRes(prev_frame.cols, prev_frame.rows);

    // project points to image coordinates
    for (int i = 0; i < imageCount; i++) {
        // project
        cameraKMatrix.push_back(K);
        distCoeffsMatrix.push_back(distCoeffs);

    }

    cout << "Bundle Adjustment" << endl;
    int M = reconstructCloudVec.size();
    int N = imageCount;

    vector<vector<Point2d> > imagePoints(N,vector<Point2d>(M)); // projections of 3d points for every camera
    vector<vector<int> > visibility(N,vector<int>(M, 0)); // visibility of 3d points for every camera

    for (int pt3d = 0; pt3d < reconstructCloudVec.size(); pt3d++) {
        for (int i = 0; i < imageCount; i++) {
            imagePoints[i][pt3d] = img_pt_matches_pts[pt3d][i];
            if(img_pt_matches[pt3d][i] >= 0){
                visibility[i][pt3d] = 1;
            }

        }
    }

//    cv::LevMarqSparse lms;
//    lms.bundleAdjust(points_opt, imagePoints, visiblity, cameraMatrix, R_opt, T_opt, distCoeffs, criteria);
/*Params ( TYPE t= MOTIONSTRUCTURE, int iters = 150,double minErr = 1e-10,
            int  fixedIntri =5,int  fixedDist = 5,bool Verbose=false )*/
    cvsba::Sba sba;
    cvsba::Sba::Params params = cvsba::Sba::Params ( cvsba::Sba::MOTIONSTRUCTURE, 150,1e-6,5,5,false );
    sba.setParams(params);
    sba.run(reconstructCloudVec,  imagePoints,  visibility,  cameraKMatrix,  R_opt,  T_opt, distCoeffsMatrix);
    std::cout<<"Initial error="<<sba.getInitialReprjError()<<". Final error="<<sba.getFinalReprjError()<<std::endl;

    Mat final = reshape.t();
    savePointList("projectiveReconstructionBeforeBA.asc", final);

    savePointList("projectiveReconstructionAfterBA.asc", reconstructCloudVec);

#ifdef PCL
// set up viewer
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0);

    for (int i = 0; i < imageCount; i++) {
        cv::Mat Rw, Tw;
        Eigen::Matrix4f _t;
        Eigen::Affine3f t;

        // initial camera coordinate
        Rw = R_opt[i].t();
        Tw = -Rw * T_opt[i];

        _t << Rw.at<double>(0, 0), Rw.at<double>(0, 1), Rw.at<double>(0, 2), Tw.at<double>(0, 0),
                Rw.at<double>(1, 0), Rw.at<double>(1, 1), Rw.at<double>(1, 2), Tw.at<double>(1, 0),
                Rw.at<double>(2, 0), Rw.at<double>(2, 1), Rw.at<double>(2, 2), Tw.at<double>(2, 0),
                0.0, 0.0, 0.0, 1.0;

        t = _t;
        viewer.addCoordinateSystem(3.0, t, 0);
    }
    viewer.initCameraParameters();


// Fill in the true point data
    pcl::PointCloud<pcl::PointXYZRGB> cloud_true;
    cloud_true.width = prev_xp_p.size();
    cloud_true.height = 1;
    cloud_true.is_dense = false;
    cloud_true.points.resize(cloud_true.width * cloud_true.height);

    for (size_t i = 0; i < cloud_true.points.size(); ++i) {
        cloud_true.points[i].x = prev_xp_p[i].x;
        cloud_true.points[i].y = prev_xp_p[i].y;
        cloud_true.points[i].z = prev_xp_p[i].z;
//        cloud_true.points[i].r = images[prevIdx].at<Vec3b>(prev_points_raw[i]).val[0];
//        cloud_true.points[i].g = images[prevIdx].at<Vec3b>(prev_points_raw[i]).val[1];
//        cloud_true.points[i].b = images[prevIdx].at<Vec3b>(prev_points_raw[i]).val[2];
        cloud_true.points[i].r = 255;
        cloud_true.points[i].g = 255;
        cloud_true.points[i].b = 255;
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> green(cloud_true.makeShared());


//    // add points
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud_true.makeShared(), green, "true points");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "true points");

    viewer.spin();

#endif

    if (waitKey(0)) {
        return 0;
    }

    return 0;
}






