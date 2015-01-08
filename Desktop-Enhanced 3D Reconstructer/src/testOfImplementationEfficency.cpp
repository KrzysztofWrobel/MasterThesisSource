/* Example 10-1. Pyramid Lucas-Kanade optical flow code

 ************************************************** */
#include <iostream>
#include <fstream>
#include <cassert>
#include <exception>
#include <sstream>
#include <string>
#include <boost/format.hpp>

#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Utils64.h"
#include "IOUtils.h"
#include "TestUtils.h"
#include "FeatureMatching.h"
#include "MultiView.h"
#include "src/five-point-nister/_modelest.h"
#include <src/five-point-nister/five-point.hpp>

#include <stdio.h>
#include <time.h>

#define DEBUG
#define TIME_DEBUG

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

    cout << "Please enter Sift features number: ";
    int SIFT_FEATURES = 200;
    cin >> SIFT_FEATURES;

    std::vector<cv::Mat> images;
    std::vector<uchar> status, status1;
    vector<KeyPoint> prev_keypoints;
    vector<KeyPoint> next_keypoints;
    Mat prev_descriptors;
    Mat next_descriptors;
    vector<DMatch> good_matches;
    vector<Point2d> points1;
    vector<Point2d> points2;
    vector<Point2d> points1En;
    vector<Point2d> points2En;
    vector<Point2d> points1Essen;
    vector<Point2d> points2Essen;
    vector<Point3d> points1EssenExp;
    vector<Point3d> points2EssenExp;
    vector<Point2d> points1St;
    vector<Point2d> points2St;

    vector<Point2d> prev_points_raw; //Raw points from Keypoints
    vector<Point2d> next_points_raw;

    Mat K, distCoeffs;
    double timeF, timeFEn, time3point,timeEssen, timeEssenEn, timeNone;
    std::chrono::high_resolution_clock::time_point tStart, tEnd;

    cv::FileStorage fs;
    fs.open("out_camera_data.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> distCoeffs;
    K.convertTo(K, CV_64FC1);
    distCoeffs.convertTo(distCoeffs, CV_64FC1);
    cout << K << endl;
    cout << K << endl;

    std::vector<ImageDesc> imageDescriptions;
    char const *filename = "sensor.txt";
    parseSensorFile(filename, imageDescriptions);

    for (int i = 0; i < imageDescriptions.size(); i++) {
        images.push_back(imread(imageDescriptions[i].path, CV_LOAD_IMAGE_COLOR));   // Read the file
    }

    int prevIdx = 0;
    int nextIdx = 1;

    Mat prev_frame, next_frame;
    cvtColor(images[prevIdx], prev_frame, CV_RGB2GRAY);
    cvtColor(images[nextIdx], next_frame, CV_RGB2GRAY);
    GaussianBlur(prev_frame, prev_frame, Size(3, 3), 1.5, 1.5);
    GaussianBlur(next_frame, next_frame, Size(3, 3), 1.5, 1.5);

#ifdef TIME_DEBUG
    std::chrono::high_resolution_clock::time_point tMatchesSt = std::chrono::high_resolution_clock::now();
#endif

    getSiftKeypointsAndDescriptors(prev_frame, prev_keypoints, prev_descriptors, SIFT_FEATURES);
    getSiftKeypointsAndDescriptors(next_frame, next_keypoints, next_descriptors, SIFT_FEATURES);
    getCorrespondingPoints(prev_keypoints, next_keypoints, prev_descriptors, next_descriptors, good_matches, prev_points_raw, next_points_raw);

#ifdef TIME_DEBUG
    std::chrono::high_resolution_clock::time_point tMatchesEnd = std::chrono::high_resolution_clock::now();
    double durationMatches = std::chrono::duration_cast<std::chrono::microseconds>(tMatchesEnd - tMatchesSt).count();
    cout << "Matches Duration(ms): " << durationMatches << endl;
#endif

    Mat rotDiffGlobal;
    double pitchDiff, azimuthDiff, rollDiff;
    getRelativeMatrixAndAngles(imageDescriptions, prevIdx, nextIdx, rotDiffGlobal, pitchDiff, azimuthDiff, rollDiff);

    Mat prev_rot, next_rotation;
    points1 = prev_points_raw;
    points2 = next_points_raw;
    points1En = prev_points_raw;
    points2En = next_points_raw;
    points1Essen = prev_points_raw;
    points2Essen = next_points_raw;
    points1St = prev_points_raw;
    points2St = next_points_raw;
    convertPointsToHomogeneous(prev_points_raw, points1EssenExp);
    convertPointsToHomogeneous(next_points_raw, points2EssenExp);


    double minVal1, maxVal1;
    int fundEssenEstimationMethod = FM_RANSAC; //FM_RANSAC FM_LMEDS

//    FindFundamentalOpenCVEnhanced
    Mat dR;
    Mat TProp;
    tStart = std::chrono::high_resolution_clock::now();
    Mat TdRExp;
    for(int i = 0; i < 100; i++) {
        TdRExp = findFundamentalEnhanced(prev_points_raw, next_points_raw, K, distCoeffs, fundEssenEstimationMethod, rotDiffGlobal, dR, TProp, status);
    }
    tEnd = std::chrono::high_resolution_clock::now();
    timeFEn = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

    Mat points1ess5(points1Essen.size(), 3, CV_64FC1), points2ess5(points2Essen.size(), 3, CV_64FC1);
    for (int a = 0; a < points1Essen.size(); a++) {
        points1ess5.at<double>(a, 0) = points1Essen[a].x;
        points1ess5.at<double>(a, 1) = points1Essen[a].y;
        points1ess5.at<double>(a, 2) = 1;
        points2ess5.at<double>(a, 0) = points2Essen[a].x;
        points2ess5.at<double>(a, 1) = points2Essen[a].y;
        points2ess5.at<double>(a, 2) = 1;
    }
    CV_Assert(points1ess5.type() == CV_64FC1);
    cv::minMaxIdx(points1ess5, &minVal1, &maxVal1);

    tStart = std::chrono::high_resolution_clock::now();
    Mat EssentialReal;
    for(int i = 0; i < 100; i++) {
        EssentialReal = findEssentialMat(points1EssenExp, points2EssenExp, K.at<double>(0), Point2d(K.at<double>(0, 2), K.at<double>(1, 2)), fundEssenEstimationMethod, 0.99, 0.006 * maxVal1, status1); //threshold from [Snavely07 4.1]
    }
    tEnd = std::chrono::high_resolution_clock::now();
    timeEssen = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
    cout << "essential keeping " << countNonZero(status1) << " / " << status1.size() << endl;
    for (int i = status1.size() - 1; i >= 0; i--) {
        if (!status1[i]) {
            points1EssenExp.erase(points1EssenExp.begin() + i);
            points2EssenExp.erase(points2EssenExp.begin() + i);
        }
    }

    tStart = std::chrono::high_resolution_clock::now();
    Mat essentialMatEnhanced;
    for(int i = 0; i < 100; i++) {
        essentialMatEnhanced = findEssentialMatEnhanced(points1ess5, points2ess5, K, rotDiffGlobal, fundEssenEstimationMethod, status1);
    }
    tEnd = std::chrono::high_resolution_clock::now();
    timeEssenEn = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

    cout << "essential enhanced keeping " << countNonZero(status1) << " / " << status1.size() << endl;
    for (int i = status1.size() - 1; i >= 0; i--) {
        if (!status1[i]) {
            points1Essen.erase(points1Essen.begin() + i);
            points2Essen.erase(points2Essen.begin() + i);
        }
    }

    double minVal, maxVal;
    cv::minMaxIdx(points1, &minVal, &maxVal);

    std::chrono::high_resolution_clock::time_point ti1 = std::chrono::high_resolution_clock::now();
    tStart = std::chrono::high_resolution_clock::now();
    Mat F;
    for(int i = 0; i < 100; i++) {
        F = findFundamentalMat(points1, points2, fundEssenEstimationMethod, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
        F.convertTo(F, CV_64FC1);
        Mat E = K.t() * F * K;

        Mat R, R1;
        Mat t;
        decomposeEssentialMat(E, R, R1, t);
    }
    tEnd = std::chrono::high_resolution_clock::now();
    timeF = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

    std::chrono::high_resolution_clock::time_point ti2 = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(ti2 - ti1).count();
    cout << "Duration: " << duration << endl;

    cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;
    for (int i = status.size() - 1; i >= 0; i--) {
        if (!status[i]) {
            points1.erase(points1.begin() + i);
            points2.erase(points2.begin() + i);
            good_matches.erase(good_matches.begin() + i);
        }
    }

    Mat img_matches;
    drawMatches(prev_frame, prev_keypoints, next_frame, next_keypoints, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", img_matches);


    Mat FEnhanced;
    Mat tEnhanced;
    uchar *goodStatuses;

    tStart = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++) {
        std::chrono::high_resolution_clock::time_point tRansacEnStart = std::chrono::high_resolution_clock::now();
        getTranslationWithKnownRotation(prev_points_raw, next_points_raw, K, rotDiffGlobal, FEnhanced, tEnhanced, goodStatuses);
        std::chrono::high_resolution_clock::time_point tRansacEnEnd = std::chrono::high_resolution_clock::now();
        double durationRansacEn = std::chrono::duration_cast<std::chrono::microseconds>(tRansacEnEnd - tRansacEnStart).count();
    }
//    cout << "Duration ransac enhanced: " << durationRansacEn << endl;
    tEnd = std::chrono::high_resolution_clock::now();
    time3point = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

    //Correction check
//    Mat corrRotDiffGlobal = dR * rotDiffGlobal;
//    getTranslationWithKnownRotation(prev_points_raw, next_points_raw, K, corrRotDiffGlobal, FEnhanced, tEnhanced, goodStatuses);

    for (int i = prev_points_raw.size() - 1; i >= 0; i--) {
        if (!goodStatuses[i]) {
            points1En.erase(points1En.begin() + i);
            points2En.erase(points2En.begin() + i);
        }
    }

    Mat FStandard;
    Mat tStandard;

    tStart = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point tRansacStart = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++) {
        findFundamentalStandard(prev_points_raw, next_points_raw, FStandard, goodStatuses);
    }
    tEnd = std::chrono::high_resolution_clock::now();
    timeF = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
//    Mat EStd = K.t() * FStandard * K;
//    Mat Rx, R1x, tx;
//    decomposeEssentialMat(EStd, Rx, R1x, tx);
//    Mat t1x = -tx;

    std::chrono::high_resolution_clock::time_point tRansacEnd = std::chrono::high_resolution_clock::now();
    double durationRansac = std::chrono::duration_cast<std::chrono::microseconds>(tRansacEnd - tRansacStart).count();
    cout << "Duration ransac standard: " << durationRansac << endl;

    for (int i = prev_points_raw.size() - 1; i >= 0; i--) {
        if (!goodStatuses[i]) {
            points1St.erase(points1St.begin() + i);
            points2St.erase(points2St.begin() + i);
        }
    }

    Mat FEssential = K.inv().t() * (essentialMatEnhanced * rotDiffGlobal) * K.inv();
    FEssential = FEssential / FEssential.at<double>(8);

    Mat FEssentialNor = K.inv().t() * (EssentialReal) * K.inv();
    FEssentialNor = FEssentialNor / FEssentialNor.at<double>(8);

    cout << "TdRExp" << TdRExp << endl;
    Mat FdR = K.inv().t() * (TdRExp * rotDiffGlobal) * K.inv();
    FdR = FdR / FdR.at<double>(8);

    tStart = std::chrono::high_resolution_clock::now();
    Mat Treal = Mat(3, 1, CV_64FC1);
    Treal.at<double>(0) = (imageDescriptions[nextIdx].globalPosX - imageDescriptions[prevIdx].globalPosX);
    Treal.at<double>(1) = (imageDescriptions[nextIdx].globalPosY - imageDescriptions[prevIdx].globalPosY);
    Treal.at<double>(2) = (imageDescriptions[nextIdx].globalPosZ - imageDescriptions[prevIdx].globalPosZ);
//    Treal = -rotDiffGlobal * Treal;
    Mat FReal = K.inv().t() * (makeSkewMatrix(Treal) * rotDiffGlobal) * K.inv();
    FReal = FReal / FReal.at<double>(8);
    tEnd = std::chrono::high_resolution_clock::now();
    timeNone = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();

    //TODO none estimation
    float test = 0, test1 = 0, test2 = 0, test3 = 0, test4 = 0, test6 = 0;
    for (int i = 0; i < points1.size(); i++) {
        test += sampson_error((double *) F.data, points1[i].x, points1[i].y, points2[i].x, points2[i].y);
//        test1 += sampson_error((double *) FStandard.data, points1St[i].x, points1St[i].y, points2St[i].x, points2St[i].y);
        test1 += sampson_error((double *) FReal.data, points1[i].x, points1[i].y, points2[i].x, points2[i].y);
        test1 += sampson_error((double *) FStandard.data, points1St[i].x, points1St[i].y, points2St[i].x, points2St[i].y);
        test3 += sampson_error((double *) FEnhanced.data, points1En[i].x, points1En[i].y, points2En[i].x, points2En[i].y);
        test6 += sampson_error((double *) FdR.data, points1[i].x, points1[i].y, points2[i].x, points2[i].y);
        test2 += sampson_error((double *) FEssential.data, points1Essen[i].x, points1Essen[i].y, points2Essen[i].x, points2Essen[i].y);
        test4 += sampson_error((double *) FEssentialNor.data, points1Essen[i].x, points1Essen[i].y, points2Essen[i].x, points2Essen[i].y);
    }
    cout << "Test ransac from OpenCv: " << test << endl;
    cout << "Test enhanced: " << test3 << endl;
    cout << "Test FdR: " << test6 << endl;
    cout << "Test known rotation and trans: " << test1 << endl;
    cout << "Test ransac essential: " << test2 << endl;
    cout << "Test ransac essentialEnhanced: " << test4 << endl;

    cout << "Test ransac from OpenCv(per point): " << test / points1.size() << " for " << points1.size() << " points" << endl;
    cout << "Test enhanced(per point): " << test3 / points1En.size() << " for " << points1En.size() << " points" << endl;
    cout << "Test FdR(per point): " << test6 / points1.size() << " for " << points1.size() << " points" << endl;
    cout << "Test known rotation and trans(per point): " << test1 / points1St.size() << " for " << points1St.size() << " points" << endl;
    cout << "Test ransac essential(per point): " << test2 / points1Essen.size() << "for " << points1Essen.size() << " points" << endl;
    cout << "Test ransac essentialEnhanced(per point): " << test4 / points1Essen.size() << " for " << points1Essen.size() << " points" << endl;


    cout << boost::format("Test ransac from OpenCv: %.2f") % (timeF/1000) << endl;
    cout << boost::format("Test enhanced: %.2f") % (timeFEn/1000) << endl;
    cout << boost::format("Test FdR: %.2f") % (time3point/1000) << endl;
    cout << boost::format("Test known rotations: %.2f") %  (timeNone/1000) << endl;
    cout << boost::format("Test ransac essential: %.2f") % (timeEssen/1000) << endl;
    cout << boost::format("Test ransac essentialEnhanced: %.2f") % (timeEssenEn/1000) << endl;

    int cols = prev_frame.cols;
    int rows = prev_frame.rows;
    cv::Mat outImg1(rows * 3, cols * 2, CV_8UC3);
    cv::Mat outImg2(rows * 3, cols * 2, CV_8UC3);


    drawEpipolarLines<double, double>("Epipolars OpenCV", F, prev_frame, next_frame, points1, points2, outImg1, Rect(0, 0, 2*cols, rows), -1);
//    drawEpipolarLines<double, double>("Epipolars Standard", FStandard, prev_frame, next_frame, points1St, points2St, outImg, Rect(2*cols, 0,  4 * cols, rows), -1);
    drawEpipolarLines<double, double>("Epipolars Fdr - ENhanced", FdR, prev_frame, next_frame, points1, points2, outImg1, Rect(0, rows, 2 * cols, rows), -1);
    drawEpipolarLines<double, double>("Epipolars Enhanced", FEnhanced, prev_frame, next_frame, points1En, points2En, outImg1, Rect(0, 2* rows,2* cols, rows), -1);
    drawEpipolarLines<double, double>("Epipolars Real", FReal, prev_frame, next_frame, points1, points2, outImg2, Rect(0, 0,  2 * cols, rows), -1);
    drawEpipolarLines<double, double>("Epipolars Essential", FEssentialNor, prev_frame, next_frame, points1Essen, points2Essen, outImg2, Rect(0, rows, 2*cols, rows), -1);
    drawEpipolarLines<double, double>("Epipolars Essential Enhanced", FEssential, prev_frame, next_frame, points1Essen, points2Essen, outImg2, Rect(0, 2 * rows, 2* cols, rows) ,- 1);
    imwrite("Summary1.jpg", outImg1);
    imwrite("Summary2.jpg", outImg2);

    if (waitKey(300000)) {
        return 0;
    }

    return 0;
}







