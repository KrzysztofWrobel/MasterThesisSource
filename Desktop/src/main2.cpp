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

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "Utils64.h"
#include "IOUtils.h"
#include "TestUtils.h"
#include "FeatureMatching.h"
#include "MultiView.h"
#include "modelest.h"

#include <stdio.h>
#include <time.h>

#define DEBUG
#define TEST
#define MAX_CORNERS 100
#define SIFT_FEATURES 3000

using namespace std;
using namespace cv;


int main(int argc, char **argv) {

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> cameraKMatrix;
    std::vector<cv::Mat> R_true, R_init, R_opt;
    std::vector<cv::Mat> T_true, T_init, T_opt;
    std::vector<cv::Mat> cameraDistCoeffs;
    std::vector<std::vector<int> > visiblity;


    int N = 11;
    int numCamera = N;
    cv::Size cameraRes(640, 480);
    cv::Mat cameraMat = (cv::Mat_<double>(3, 3) <<
            cameraRes.width / 4, 0, cameraRes.width / 2,
            0, cameraRes.height / 4, cameraRes.height / 2,
            0, 0, 1);

    cv::RNG rng(0);

    std::vector<ImageDesc> imageDescriptions;
    char const *sensorDataFilename = "sensor.txt";
    parseSensorFile(sensorDataFilename, imageDescriptions);

    Mat K, distCoeffs;
    char const *cameraIntersincFilename = "out_camera_data.yml";
    parseCameraInstersincParameters(cameraIntersincFilename, K, distCoeffs);

    // variables for sba
    std::vector<cv::Point3d> points_true, points_init, points_opt;
    std::vector<std::vector<cv::Point2d> > imagePoints;

    // generate points
    for (double y = -10; y < 10; y += 1) {
        double r = 4 + sin(3 * M_PI * y / 10);
        for (double theta = 0; theta <= 2 * M_PI; theta += M_PI / 4) {
            cv::Point3d point_true(r * cos(theta) + y /*+ rng.gaussian(1)*/,
                    5 - y /*+ rng.gaussian(1)*/,
                    r * sin(theta) + y /*+ rng.gaussian(1)*/);
            cv::Point3d point_init(point_true.x + rng.gaussian(5),
                    point_true.y + rng.gaussian(5),
                    point_true.z + rng.gaussian(5));
            points_true.push_back(point_true);
            points_init.push_back(point_init);
            points_opt.push_back(point_init);
        }
    }

    // define cameras
    for (int i = 0; i < numCamera; i++) {
        cameraKMatrix.push_back(cameraMat);
        cameraDistCoeffs.push_back((cv::Mat(5,1,CV_64FC1, cv::Scalar::all(0))));

        cv::Mat _R_true(3, 3, CV_64FC1), _R_init(3, 3, CV_64FC1);
        _R_true = (cv::Mat_<double>(3, 3) <<
                cos(i * 2 * M_PI / N), 0, sin(i * 2 * M_PI / N),
                0, 1, 0,
                -sin(i * 2 * M_PI / N), 0, cos(i * 2 * M_PI / N));
//        rng.fill(_R_init, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(0.5));
        _R_init += _R_true;
        R_true.push_back(_R_true);
        R_init.push_back(_R_init);
        R_opt.push_back(_R_init);

        cv::Mat _T_true(3, 1, CV_64FC1), _T_init(3, 1, CV_64FC1);
        _T_true = (cv::Mat_<double>(3, 1) << 0, 0, 30);
//        rng.fill(_T_init, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(0.5));
        _T_init += _T_true;
        T_true.push_back(_T_true);
        T_init.push_back(_T_init);
        T_opt.push_back(_T_init);
    }

    // project points to image coordinates
    for (int i = 0; i < cameraKMatrix.size(); i++) {
        // project
        std::vector<cv::Point2d> imagePoint;
        cv::projectPoints(points_true, R_true[i], T_true[i], cameraKMatrix[i], cameraDistCoeffs[i], imagePoint);

        // check if the point is in cameras
        std::vector<int> vis(points_true.size(), 0);
        for (int j = 0; j < imagePoint.size(); j++) {
            // if the image point is within camera resolution then the point is visible
            if ((0 <= imagePoint[j].x) && (imagePoint[j].x <= cameraRes.width) &&
                    (0 <= imagePoint[j].y) && (imagePoint[j].y <= cameraRes.height) &&
                    (rng.uniform(-1, 9))) { // add randomness

                // perturbate
                vis[j] = 1;
//                imagePoint[j].x += rng.gaussian(3);
//                imagePoint[j].y += rng.gaussian(3);

            }
                // else, the point is not visible
            else {

                vis[j] = 0;
                imagePoint[j].x = -1;
                imagePoint[j].y = -1;
            }
        }
        imagePoints.push_back(imagePoint);
        visiblity.push_back(vis);
    }

//IMPORTANT: most of the mobile cameras have 1:1 ratio so we can actually load it ourselves
//    Loaded camera matrix: [6552.2197265625, 0, 639.5;
//    0, 6552.2197265625, 359.5;
//    0, 0, 1]
//    Loaded distortion coefficients: [-15.25522613525391; 5914.3701171875; 0; 0; 60.90658187866211]
    K.at<double>(0, 0) = 1280;
    K.at<double>(0, 1) = 0;
    K.at<double>(1, 1) = 1280;
    K.at<double>(0, 2) = 639.5;
    K.at<double>(1, 2) = 359.5;
#ifdef DEBUG
    cout << "K: " << K << endl;
#endif

    Mat prev_frame, next_frame;
    vector<KeyPoint> prev_keypoints, next_keypoints;
    Mat prev_descriptors, next_descriptors;

    std::vector<uchar> status, status1, trackedStatus;
    std::vector<double> err;

    vector<DMatch> good_matches;
    vector<Point2d> points1En, points2En;
    vector<Point2d> points1St, points2St;
    vector<Point2d> prev_points_raw, next_points_raw;


    int imageCount = imageDescriptions.size();
    for (int i = 0; i < imageCount; i++) {
        images.push_back(imread(imageDescriptions[i].path, CV_LOAD_IMAGE_COLOR));   // Read the file
    }

    int startIdx = 0;
//    int endIdx = imageCount - 1;
    int endIdx = 10;
    cvtColor(images[startIdx], prev_frame, CV_RGB2GRAY);
    GaussianBlur(prev_frame, prev_frame, Size(3, 3), 1.5, 1.5);
    getSiftKeypointsAndDescriptors(prev_frame, prev_keypoints, prev_descriptors, SIFT_FEATURES);

    double prevPitch, prevAzimuth, prevRoll;
    double nextPitch, nextAzimuth, nextRoll;
    double pitchDiff, azimuthDiff, rollDiff;
    Mat prev_rot, next_rotation;
    Mat prev_trans, next_trans;

    prevPitch = (imageDescriptions[startIdx].pitch);
    prevAzimuth = (imageDescriptions[startIdx].azimuth);
    prevRoll = (imageDescriptions[startIdx].roll);

//    prev_rot = getRotated3DMatrix<double>(radians(prevPitch), radians(prevAzimuth), radians(prevRoll));
    prev_rot = Mat::eye(3, 3, CV_64FC1);
    prev_trans = Mat::zeros(3, 1, CV_64FC1);

    Mat reconstructCloud = Mat::zeros(0, 4, CV_64FC1);

    for (int nextIdx = startIdx + 1; nextIdx <= endIdx; nextIdx++) {
        int prevIdx = nextIdx - 1;

#ifndef TEST
        cvtColor(images[nextIdx], next_frame, CV_RGB2GRAY);
        GaussianBlur(next_frame, next_frame, Size(3, 3), 1.5, 1.5);
        getSiftKeypointsAndDescriptors(next_frame, next_keypoints, next_descriptors, SIFT_FEATURES);

        getCorrespondingPoints(prev_keypoints, next_keypoints, prev_descriptors, next_descriptors, good_matches, prev_points_raw, next_points_raw);

        nextPitch = (imageDescriptions[nextIdx].pitch);
        nextAzimuth = (imageDescriptions[nextIdx].azimuth);
        nextRoll = (imageDescriptions[nextIdx].roll);

        pitchDiff = radians((nextPitch - prevPitch));
        azimuthDiff = radians((nextAzimuth - prevAzimuth));
        rollDiff = radians(-(nextRoll - prevRoll)); //TODO Due to Android app bug we need to minus this one
//        Mat rotDiffGlobal = getRotated3DMatrix<double>(pitchDiff, azimuthDiff, rollDiff);
        Mat rotDiffGlobal = getRotated3DMatrix<double>(pitchDiff, azimuthDiff, rollDiff);

//        Mat prev_rot = imageDescriptions[prevIdx].rotationMatrix;
//        Mat next_rotation = imageDescriptions[nextIdx].rotationMatrix;
//    Mat next_rotation = getRotated3DMatrix<double>(radians(pitch2), radians(azimuth2), radians(roll2));
        next_rotation = rotDiffGlobal * prev_rot;
#else
        prev_rot = R_true[prevIdx];
        next_rotation = R_true[nextIdx];
        Mat rotDiffGlobal = prev_rot.inv() * next_rotation;

        prev_points_raw = imagePoints[prevIdx];
        next_points_raw = imagePoints[nextIdx];
#endif
        points1En = prev_points_raw;
        points2En = next_points_raw;

        double minVal, maxVal;
        cv::minMaxIdx(prev_points_raw, &minVal, &maxVal);

        Mat FEnhanced;
        Mat tEnhanced;
        const int maxIterations = 20;
        double reprojThreshold = 0.006 * maxVal;
        const int count = prev_points_raw.size();

        //Enhanced reconstruction
        {
            vector<Point2d> point1s;
            vector<Point2d> point2s;

            //Starting RANSAC for new algorithm
            uchar *goodStatuses = new uchar[count];
            float *errors = new float[count];
            uchar *statuses = new uchar[count];

            int iterationNumber = 0;
            int niters = maxIterations;
            const int modelPoints = 3;


            int maxGoodCount = 0;
            const double confidence = 0.99;

            std::chrono::high_resolution_clock::time_point tRansacStart = std::chrono::high_resolution_clock::now();

            for (iterationNumber = 0; iterationNumber < niters; iterationNumber++) {

                getSubset(prev_points_raw, next_points_raw, point1s, point2s, 300, modelPoints);

                //TODO how to decide it's minus or not
                Mat t = findTranslation(point1s, point2s, rotDiffGlobal, K);
                Mat F1 = constructFundamentalMatrix(rotDiffGlobal, t, K);

                int goodCount = findInliers(prev_points_raw, next_points_raw, F1, errors, statuses, reprojThreshold);
                if (goodCount > maxGoodCount) {
                    std::swap(statuses, goodStatuses);
                    FEnhanced = F1;
                    tEnhanced = t;
                    maxGoodCount = goodCount;
                    niters = cvRANSACUpdateNumIters(confidence,
                            (double) (count - maxGoodCount) / count, modelPoints, niters);

                }

            }
            std::chrono::high_resolution_clock::time_point tRansacEnd = std::chrono::high_resolution_clock::now();
            double durationRansac = std::chrono::duration_cast<std::chrono::milliseconds>(tRansacEnd - tRansacStart).count();
            cout << "Duration ransac enhanced: " << durationRansac << endl;

            cout << "Good count: " << maxGoodCount << endl;
            cout << "Translation: " << tEnhanced << endl;
            cout << "FEnhanced: " << FEnhanced << endl;

            for (int i = count - 1; i >= 0; i--) {
                if (!goodStatuses[i]) {
                    points1En.erase(points1En.begin() + i);
                    points2En.erase(points2En.begin() + i);
                }
            }
        }

#ifdef DEBUG
        cout << "Points1: " << points1En.size()<<endl;
        cout << "Points2: " << points2En.size()<<endl;
//    drawEpipolarLines<double, double>("EpipolarsEnhanced", FEnhanced, prev_frame, next_frame, points1En, points2En, -1);
#endif

        Matx34d prev_PEnhanced, next_PEnhanced;
        tEnhanced = -tEnhanced;

        Mat prev_transy = Mat::zeros(3, 1, CV_64FC1);
        next_trans = prev_trans - (tEnhanced);
        cout << "Prev_trans: " << prev_trans << endl;
        cout << "Next_trans: " << next_trans << endl;
        R_init[prevIdx] = prev_rot;
        R_init[nextIdx] = next_rotation;
        T_init[prevIdx] = prev_trans;
        T_init[nextIdx] = next_trans;

        prev_PEnhanced = Matx34d(prev_rot.at<double>(0, 0), prev_rot.at<double>(0, 1), prev_rot.at<double>(0, 2), prev_transy.at<double>(0),
                prev_rot.at<double>(1, 0), prev_rot.at<double>(1, 1), prev_rot.at<double>(1, 2), prev_transy.at<double>(1),
                prev_rot.at<double>(2, 0), prev_rot.at<double>(2, 1), prev_rot.at<double>(2, 2), prev_transy.at<double>(2));
        next_PEnhanced = Matx34d(next_rotation.at<double>(0, 0), next_rotation.at<double>(0, 1), next_rotation.at<double>(0, 2), (tEnhanced).at<double>(0),
                next_rotation.at<double>(1, 0), next_rotation.at<double>(1, 1), next_rotation.at<double>(1, 2), (tEnhanced).at<double>(1),
                next_rotation.at<double>(2, 0), next_rotation.at<double>(2, 1), next_rotation.at<double>(2, 2), (tEnhanced).at<double>(2));


        undistortPoints(points1En, points1En, K, distCoeffs);
        undistortPoints(points2En, points2En, K, distCoeffs);

        vector<Point3d> pt_3d1;
        Mat XPenhanced = Mat::zeros(4, 0, CV_64FC1);

        triangulatePoints(prev_PEnhanced, next_PEnhanced, points1En, points2En, XPenhanced);

        Mat translate = Mat::eye(4, 4, CV_64FC1);
        translate.at<double>(0,3) = prev_trans.at<double>(0);
        translate.at<double>(1,3) = prev_trans.at<double>(1);
        translate.at<double>(2,3) = prev_trans.at<double>(2);
        translate.at<double>(3,3) = 1;
        XPenhanced = translate * XPenhanced;
        convertPointsFromHomogeneous(XPenhanced.reshape(4, 1), pt_3d1);

        TestTriangulation(pt_3d1, next_PEnhanced, status);
        status.clear();

        //End of the loop
        prevPitch = nextPitch;
        prevAzimuth = nextAzimuth;
        prevRoll = nextRoll;

        prev_rot = next_rotation.clone();
        prev_trans = next_trans.clone();
        prev_frame = next_frame;
        prev_keypoints = next_keypoints;
        prev_descriptors = next_descriptors;

        reconstructCloud.push_back(Mat(XPenhanced.t()));
    }

//    cout << reconstructCloud << endl;
    reconstructCloud = reconstructCloud.t();
    savePointList("projectiveReconstructionEnhanced.asc", reconstructCloud);
    vector<Point3d> pt_3d1;
    convertPointsFromHomogeneous(reconstructCloud.t(), pt_3d1);

    // set up viewer
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(2.0, 0);
    viewer.initCameraParameters();

    // Fill in the true point data
    pcl::PointCloud<pcl::PointXYZRGB> cloud_true;
    cloud_true.width = points_true.size();
    cloud_true.height = 1;
    cloud_true.is_dense = false;
    cloud_true.points.resize(cloud_true.width * cloud_true.height);

    for (size_t i = 0; i < cloud_true.points.size(); ++i) {
        cloud_true.points[i].x = points_true[i].x;
        cloud_true.points[i].y = points_true[i].y;
        cloud_true.points[i].z = points_true[i].z;
        cloud_true.points[i].r = 0;
        cloud_true.points[i].g = 255;
        cloud_true.points[i].b = 0;
    }
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> green(cloud_true.makeShared());

    // Fill in the initial point data
    pcl::PointCloud<pcl::PointXYZRGB> cloud_init;
    cloud_init.width = pt_3d1.size();
    cloud_init.height = 1;
    cloud_init.is_dense = false;
    cloud_init.points.resize(cloud_init.width * cloud_init.height);

    for (size_t i = 0; i < pt_3d1.size(); ++i) {
        cloud_init.points[i].x = pt_3d1[i].x;
        cloud_init.points[i].y = pt_3d1[i].y;
        cloud_init.points[i].z = pt_3d1[i].z;
        cloud_init.points[i].r = 255;
        cloud_init.points[i].g = 255;
        cloud_init.points[i].b = 0;
    }

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> yellow(cloud_init.makeShared());

    // add points
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud_true.makeShared(), green, "true points");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "true points");

    viewer.addPointCloud<pcl::PointXYZRGB>(cloud_init.makeShared(), yellow, "initial points");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "initial points");


    // add camera coordinates
    for (int i = 0; i < cameraKMatrix.size(); i++) {
        cv::Mat Rw, Tw;
        Eigen::Matrix4f _t;
        Eigen::Affine3f t;

        // true camera coordinate
        Rw = R_true[i].t();
        Tw = -Rw * T_true[i];

        _t << Rw.at<double>(0, 0), Rw.at<double>(0, 1), Rw.at<double>(0, 2), Tw.at<double>(0, 0),
                Rw.at<double>(1, 0), Rw.at<double>(1, 1), Rw.at<double>(1, 2), Tw.at<double>(1, 0),
                Rw.at<double>(2, 0), Rw.at<double>(2, 1), Rw.at<double>(2, 2), Tw.at<double>(2, 0),
                0.0, 0.0, 0.0, 1.0;

        t = _t;
        viewer.addCoordinateSystem(2.0, t, 0);

        // initial camera coordinate
        Rw =  R_init[i].t();
        Tw = T_init[i];

        _t << Rw.at<double>(0,0), Rw.at<double>(0,1), Rw.at<double>(0,2), Tw.at<double>(0,0),
                Rw.at<double>(1,0), Rw.at<double>(1,1), Rw.at<double>(1,2), Tw.at<double>(1,0),
                Rw.at<double>(2,0), Rw.at<double>(2,1), Rw.at<double>(2,2), Tw.at<double>(2,0),
                0.0, 0.0, 0.0, 1.0;

        t = _t;
        viewer.addCoordinateSystem(3.0, t, 0);

        // optimized camera coordinate
        Rw = R_opt[i].t();
        Tw = -Rw * T_opt[i];

        _t << Rw.at<double>(0, 0), Rw.at<double>(0, 1), Rw.at<double>(0, 2), Tw.at<double>(0, 0),
                Rw.at<double>(1, 0), Rw.at<double>(1, 1), Rw.at<double>(1, 2), Tw.at<double>(1, 0),
                Rw.at<double>(2, 0), Rw.at<double>(2, 1), Rw.at<double>(2, 2), Tw.at<double>(2, 0),
                0.0, 0.0, 0.0, 1.0;

        t = _t;
        viewer.addCoordinateSystem(5.0, t, 0);
    }

    // view
    viewer.spin();

    if (waitKey(10)) {
        return 0;
    }

    return 0;
}




