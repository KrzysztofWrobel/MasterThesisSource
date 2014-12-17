#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void defineCameras(Mat &F, Mat &P1, Mat &P2);

Mat linearTriangulation(Mat &P1, Mat &P2, vector<Point2d> &x1, vector<Point2d> &x2);

Mat findTranslation(std::vector<cv::Point2d> &points1,std::vector<cv::Point2d> &points2, Mat &rotDiff,Mat &Kinv);

Mat constructFundamentalMatrix(Mat &rotDiff, Mat &T, Mat &Kinv);

bool decideProperMatrix(Mat dRot, double tolerance);

Mat constraintMatrix(Mat dR);

bool FindPoseEstimation(
        cv::Mat &rvec,
        cv::Mat &t,
        cv::Mat &R,
        cv::Mat &K,
        cv::Mat &distCoeffs,
        std::vector<cv::Point3d> ppcloud,
        std::vector<cv::Point2d> imgPoints,
        vector<int> inliers
);

bool FindPoseEstimationEnhanced(
        cv::Mat &rvec,
        cv::Mat &t,
        cv::Mat &R,
        cv::Mat &RInit,
        cv::Mat &TInit,
        cv::Mat &K,
        cv::Mat &distCoeffs,
        std::vector<cv::Point3d> ppcloud,
        std::vector<cv::Point2d> imgPoints,
        vector<int> inliers);

Mat makeSkewMatrix(Mat &e);

Mat findFundamentalEnhanced(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &K, Mat &distCoeffs, int fundEssenEstimationMethod, Mat &rotDiffGlobal, Mat &dR, Mat &T, vector<uchar> &status);
void getTranslationWithKnownRotation(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &K, Mat &rotDiffGlobal, Mat &FEnhanced, Mat &tEnhanced, uchar *&goodStatuses);
void findFundamentalStandard(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &FStandard, uchar *&goodStatuses);

void chooseProperRAndTFromTriangulation(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &K, Mat &distCoeffs, Mat &R1, Mat &R2, Mat &t1, Mat &t2, Mat &R, Mat &T);

#define EPSILON 0.0001

Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
        Matx34d P,		//camera 1 matrix
        Point3d u1,		//homogenous image point in 2nd camera
        Matx34d P1		//camera 2 matrix
);

Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
        Matx34d P,			//camera 1 matrix
        Point3d u1,			//homogenous image point in 2nd camera
        Matx34d P1			//camera 2 matrix
);