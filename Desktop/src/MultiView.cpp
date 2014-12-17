#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "modelest.h"
#include "TestUtils.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "MultiView.h"


using namespace std;
using namespace cv;

static const int MAX_ITERS = 20;


void getEpipols(Mat &F, Mat &e1, Mat &e2) {

    SVD svd(F, SVD::FULL_UV);

    //    cout << "SVD:" << endl;
    //    cout << svd.u << endl;
    //    cout << svd.w << endl;
    //    cout << svd.vt << endl;

    float minVal = svd.w.at<double>(0, 0);
    int minIdx = 0;

    for (int i = 1; i < svd.w.rows; i++) {
        if (minVal > svd.w.at<double>(0, i)) {
            minVal = svd.w.at<double>(0, i);
            minIdx = i;
        }
    }

    //    cout << minVal << endl;
    //    cout << minIdx << endl;

    e2 = svd.u.col(minIdx).clone();
    e1 = svd.vt.row(minIdx).clone();

}

// generates skew matrix from vector
/*
 e		given vector
 return		skew matrix
 */
Mat makeSkewMatrix(Mat &e) {
    Mat sk = Mat::zeros(3, 3, CV_64FC1);
    sk.at<double>(0, 1) = -e.at<double>(2);
    sk.at<double>(0, 2) = e.at<double>(1);
    sk.at<double>(1, 0) = e.at<double>(2);
    sk.at<double>(1, 2) = -e.at<double>(0);
    sk.at<double>(2, 0) = -e.at<double>(1);
    sk.at<double>(2, 1) = e.at<double>(0);

    return sk;
}


void defineCameras(Mat &F, Mat &P1, Mat &P2) {

    P1 = Mat::eye(3, 4, CV_64FC1);

    Mat e1;
    Mat e2;
    getEpipols(F, e1, e2);
    cout << "e1 = " << e1 << endl;
    cout << "e2 = " << e2 << endl;
    Mat sk = makeSkewMatrix(e2);
    cout << "sk(e2) = " << sk << endl;

    P2 = sk * F;
    cv::hconcat(P2, e2, P2);
}

// solve homogeneous equation system by usage of SVD
/*
 A		the design matrix
 return		the estimated fundamental matrix
 */
Mat solve_dlt(Mat &A) {

    SVD svd(A, SVD::FULL_UV);

    //    cout << "SVD:" << endl;
    //    cout << svd.u << endl;
    //    cout << svd.vt << endl;

    float minVal = svd.w.at<double>(0, 0);
    int minIdx = 0;

    for (int i = 1; i < svd.w.rows; i++) {
        if (minVal > svd.w.at<double>(0, i)) {
            minVal = svd.w.at<double>(0, i);
            minIdx = i;
        }
    }

    //    cout << minVal << endl;
    //    cout << minIdx << endl;

    return svd.vt.row(minIdx).clone();
}


Mat findTranslation(std::vector<cv::Point2d> &points1, std::vector<cv::Point2d> &points2, Mat &rotDiff, Mat &Kinv) {

    Mat hg1 = Mat::zeros(points1.size(), 3, CV_64FC1);
    Mat hg2 = Mat::zeros(points2.size(), 3, CV_64FC1);
    for (int i = 0; i < points1.size(); i++) {
        hg1.at<double>(i, 0) = points1[i].x;
        hg1.at<double>(i, 1) = points1[i].y;
        hg1.at<double>(i, 2) = 1;
        hg2.at<double>(i, 0) = points2[i].x;
        hg2.at<double>(i, 1) = points2[i].y;
        hg2.at<double>(i, 2) = 1;
    }

    hg1 = hg1 * (rotDiff * Kinv).t();
    hg2 = hg2 * (Kinv).t();

    Mat A = Mat::zeros(hg1.rows, 3, CV_64FC1);
    for (int i = 0; i < hg1.rows; i++) {
        A.at<double>(i, 0) = (hg2.at<double>(i, 2) * hg1.at<double>(i, 1) - hg2.at<double>(i, 1) * hg1.at<double>(i, 2));
        A.at<double>(i, 1) = (hg2.at<double>(i, 0) * hg1.at<double>(i, 2) - hg2.at<double>(i, 2) * hg1.at<double>(i, 0));
        A.at<double>(i, 2) = (hg2.at<double>(i, 1) * hg1.at<double>(i, 0) - hg2.at<double>(i, 0) * hg1.at<double>(i, 1));
    }

    SVD svd1(A);
    Mat tCalc = svd1.vt.row(2);

    //Translation between cameras estimated and Fundamental Matrix from that as well
    Mat T = (tCalc.t());

    return T;

}

Mat constructFundamentalMatrix(Mat &rotDiff, Mat &T, Mat &Kinv) {
    Mat F1 = Kinv.t() * makeSkewMatrix(T) * rotDiff * Kinv;
    F1 = F1 / F1.at<double>(8);

    return F1;
}

Mat linearTriangulation(Mat &P1, Mat &P2, vector<Point2d> &x1, vector<Point2d> &x2) {

    Mat XP = Mat(0, 4, CV_64FC1);

    for (int i = 0; i < x1.size(); i++) {

        Mat p11 = x1[i].x * P1.row(2) - P1.row(0);
        Mat p12 = x1[i].y * P1.row(2) - P1.row(1);
        Mat p21 = x2[i].x * P2.row(2) - P2.row(0);
        Mat p22 = x2[i].y * P2.row(2) - P2.row(1);

        Mat A = Mat::zeros(0, 4, CV_64FC1);

        A.push_back(p11);
        A.push_back(p12);
        A.push_back(p21);
        A.push_back(p22);

        Mat xp = solve_dlt(A);
        XP.push_back(xp);
    }

    return XP.t();
}

bool FindPoseEstimation(
        cv::Mat &rvec,
        cv::Mat &t,
        cv::Mat &R,
        cv::Mat &K,
        cv::Mat &distCoeffs,
        std::vector<cv::Point3d> ppcloud,
        std::vector<cv::Point2d> imgPoints,
        vector<int> inliers)
{
    if (ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" << endl;
        return false;
    }

    //use CPU
    double minVal, maxVal;
    cv::minMaxIdx(imgPoints, &minVal, &maxVal);
    cv::solvePnPRansac(ppcloud, imgPoints, K, distCoeffs, rvec, t, false, 1000, 0.006 * maxVal, 0.25 * (double) (imgPoints.size()), inliers);

    vector<cv::Point2d> projected3D;
    cv::projectPoints(ppcloud, rvec, t, K, distCoeffs, projected3D);

    if (inliers.size() == 0) { //get inliers
        for (int i = 0; i < projected3D.size(); i++) {
            if (norm(projected3D[i] - imgPoints[i]) < 10.0)
                inliers.push_back(i);
        }
    }

    if (inliers.size() < (double) (imgPoints.size()) / 5.0) {
        cerr << "not enough inliers to consider a good pose (" << inliers.size() << "/" << imgPoints.size() << ")" << endl;
        return false;
    }
    cv::Rodrigues(rvec, R);

    if (cv::norm(t) > 200.0) {
        // this is bad...
        cerr << "estimated camera movement is too big, skip this camera\r\n";
        return false;
    }

    //TODO
//	if(!CheckCoherentRotation(R)) {
//		cerr << "rotation is incoherent. we should try a different base view..." << endl;
//		return false;
//	}

    std::cout << "found t = " << t << "\nR = \n" << R << std::endl;

    return true;
}

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
        vector<int> inliers)
{
    if (ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
        //something went wrong aligning 3D to 2D points..
        cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" << endl;
        return false;
    }

    cv::Rodrigues(RInit, rvec);
    t = TInit;

    //use CPU
    double minVal, maxVal;
    cv::minMaxIdx(imgPoints, &minVal, &maxVal);
    cv::solvePnPRansac(ppcloud, imgPoints, K, distCoeffs, rvec, t, true, 1000, 0.006 * maxVal, 0.25 * (double) (imgPoints.size()), inliers);

    vector<cv::Point2d> projected3D;
    cv::projectPoints(ppcloud, rvec, t, K, distCoeffs, projected3D);

    if (inliers.size() == 0) { //get inliers
        for (int i = 0; i < projected3D.size(); i++) {
            if (norm(projected3D[i] - imgPoints[i]) < 10.0)
                inliers.push_back(i);
        }
    }

    if (inliers.size() < (double) (imgPoints.size()) / 5.0) {
        cerr << "not enough inliers to consider a good pose (" << inliers.size() << "/" << imgPoints.size() << ")" << endl;
        return false;
    }
    cv::Rodrigues(rvec, R);

    if (cv::norm(t) > 200.0) {
        // this is bad...
        cerr << "estimated camera movement is too big, skip this camera\r\n";
        return false;
    }

    //TODO
//	if(!CheckCoherentRotation(R)) {
//		cerr << "rotation is incoherent. we should try a different base view..." << endl;
//		return false;
//	}

    std::cout << "found t = " << t << "\nR = \n" << R << std::endl;

    return true;
}

bool decideProperMatrix(Mat dRot, double tolerance){
    double a00 = abs(dRot.at<double>(0,0) - 1);
    double a11 = abs(dRot.at<double>(1,1) - 1);
    double a22 = abs(dRot.at<double>(2,2) - 1);
    if((a00 + a11 + a22)/3< tolerance) {
        return true;
    }else {
        return false;
    }
}

Mat constraintMatrix(Mat dR){
    Mat constDR = Mat::eye(3,3, CV_64FC1);
    double edgeVal = (dR.at<double>(0,0) + dR.at<double>(1,1) + dR.at<double>(2,2))/3;
    Mat clone = dR.clone()/edgeVal;
    constDR.at<double>(0,1) = (clone.at<double>(0,1) - clone.at<double>(1,0))/2;
    constDR.at<double>(1,0) = (clone.at<double>(1,0) - clone.at<double>(0,1))/2;
    constDR.at<double>(0,2) = (clone.at<double>(0,2) - clone.at<double>(2,0))/2;
    constDR.at<double>(2,0) = (clone.at<double>(2,0) - clone.at<double>(0,2))/2;
    constDR.at<double>(2,1) = (clone.at<double>(2,1) - clone.at<double>(1,2))/2;
    constDR.at<double>(1,2) = (clone.at<double>(1,2) - clone.at<double>(2,1))/2;

    return constDR;
}

void findFundamentalStandard(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &FStandard, uchar *&goodStatuses) {//Standard reconstruction
    {
        vector<Point2d> point1s;
        vector<Point2d> point2s;

        const int maxIterations = MAX_ITERS;
        double minVal, maxVal;
        minMaxIdx(next_points_raw, &minVal, &maxVal);
        double reprojThreshold = 0.006 * maxVal;
        const int count = prev_points_raw.size();

        //Starting RANSAC for new algorithm
        float *errors = new float[count];
        uchar *statuses = new uchar[count];
        goodStatuses = new uchar[count];

        int iterationNumber = 0;
        int niters = maxIterations;
        const int modelPoints = 8;

        int maxGoodCount = 0;
        const double confidence = 0.99;
        for (iterationNumber = 0; iterationNumber < niters; iterationNumber++) {

            getSubsety(prev_points_raw, next_points_raw, point1s, point2s, 300, modelPoints);
            Mat Fx = findFundamentalMat(point1s, point2s, FM_8POINT); //threshold from [Snavely07 4.1]

            int goodCount = findInliersy(prev_points_raw, next_points_raw, Fx, errors, statuses, reprojThreshold);
            if (goodCount > maxGoodCount) {
                swap(statuses, goodStatuses);
                FStandard = Fx;
                maxGoodCount = goodCount;
                niters = cvRANSACUpdateNumIters(confidence,
                        (double) (count - maxGoodCount) / count, modelPoints, niters);
            }
        }

        //To have more or less the same level of scene knowledge we need to include this also
        FStandard.convertTo(FStandard, CV_64FC1);
    }
}

void getTranslationWithKnownRotation(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &K, Mat &rotDiffGlobal, Mat &FEnhanced, Mat &tEnhanced, uchar *&goodStatuses) {
    vector<Point2d> point1s;
    vector<Point2d> point2s;
    const int maxIterations = MAX_ITERS;
    double minVal, maxVal;
    minMaxIdx(next_points_raw, &minVal, &maxVal);
    double reprojThreshold = 0.006 * maxVal;
    const int count = prev_points_raw.size();
    Mat Kinv = K.inv();

    //Starting RANSAC for new algorithm
    goodStatuses = new uchar[count];
    float *errors = new float[count];
    uchar *statuses = new uchar[count];

    int iterationNumber = 0;
    int niters = maxIterations;
    const int modelPoints = 3; //Important 4 gives good results for bigger matches groups

    int maxGoodCount = 0;
    const double confidence = 0.99;

    for (iterationNumber = 0; iterationNumber < niters; iterationNumber++) {

        getSubsety(prev_points_raw, next_points_raw, point1s, point2s, 300, modelPoints);
        Mat t = findTranslation(point1s, point2s, rotDiffGlobal, Kinv);
        Mat F1 = constructFundamentalMatrix(rotDiffGlobal, t, Kinv);

        int goodCount = findInliersy(prev_points_raw, next_points_raw, F1, errors, statuses, reprojThreshold);
        if (goodCount > maxGoodCount) {
            swap(statuses, goodStatuses);
            FEnhanced = F1;
            tEnhanced = t / t.at<double>(2);
            maxGoodCount = goodCount;
            niters = cvRANSACUpdateNumIters(confidence,
                    (double) (count - maxGoodCount) / count, modelPoints, niters);
        }

    }
}

Mat findFundamentalEnhanced(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &K, Mat &distCoeffs, int fundEssenEstimationMethod, Mat &rotDiffGlobal, Mat &dR, Mat &T, vector<uchar> &status) {

    vector<Point2d> points1Exp;
    vector<Point2d> points2Exp;

    points1Exp = prev_points_raw;
    points2Exp = next_points_raw;

    undistortPoints(points1Exp, points1Exp, K, distCoeffs, rotDiffGlobal);
    undistortPoints(points2Exp, points2Exp, K, distCoeffs);

    double minVal, maxVal;
    minMaxIdx(points1Exp, &minVal, &maxVal);


    Mat TdRExp = Mat(3, 3, CV_64FC1);
    TdRExp = findFundamentalMat(points1Exp, points2Exp, fundEssenEstimationMethod, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
    cout << "TdRExp keeping " << countNonZero(status) << " / " << status.size() << endl;
    TdRExp.convertTo(TdRExp, CV_64FC1);
    TdRExp /= TdRExp.at<double>(8);

    cout << TdRExp << endl;
    SVD decomp(TdRExp, SVD::FULL_UV);
    Mat W1 = Mat(Matx33d(0, -1, 0,   //HZ 9.13
            1, 0, 0,
            0, 0, 1));
    W1.convertTo(W1, CV_64FC1);
    Mat Winv2 = Mat(Matx33d(0, 1, 0,
            -1, 0, 0,
            0, 0, 1));
    Winv2.convertTo(Winv2, CV_64FC1);
    Mat dRx = decomp.u * W1 * decomp.vt; //HZ 9.19
    Mat dR1x = decomp.u * Winv2 * decomp.vt; //HZ 9.19

    cout << "dRx" << dRx << endl;
    cout << "dR1x" << dR1x << endl;

    dR = dRx;
    if (decideProperMatrix(dRx, 0.05)) {
        dR = constraintMatrix(dRx);
//        dR = dRx;
    } else if (decideProperMatrix(dR1x, 0.05)) {
        dR = constraintMatrix(dR1x);
//        dR = dR1x;
    } else if (decideProperMatrix(-dRx, 0.05)) {
        dR = constraintMatrix(-dRx);
//        dR = -dRx;
    } else if (decideProperMatrix(-dR1x, 0.05)) {
        dR = constraintMatrix(-dR1x);
//        dR = -dR1x;
    }

    Mat skewT = TdRExp * dR.inv();
    cout << "skewT" << skewT << endl;
    Mat tdecx = Mat(3,1, CV_64FC1);
    tdecx.at<double>(0) = (skewT.at<double>(2,1) - skewT.at<double>(1,2))/2;
    tdecx.at<double>(1) = (skewT.at<double>(0,2) - skewT.at<double>(2,0))/2;
    tdecx.at<double>(2) = (skewT.at<double>(1,0) - skewT.at<double>(0,1))/2;
    T = tdecx;
    cout << "dR" << dR << endl;
    cout << "tdecx" << tdecx << endl;


    return TdRExp;
}

void chooseProperRAndTFromTriangulation(vector<Point2d> &prev_points_raw, vector<Point2d> &next_points_raw, Mat &K, Mat &distCoeffs, Mat &R1, Mat &R2, Mat &t1, Mat &t2, Mat &R, Mat &T) {
    vector<Point2d> points1, points2;
    vector<Point3d> pt_3d;
    vector<uchar> status;
    Mat XP = Mat::zeros(4, 0, CV_64FC1);

    undistortPoints(prev_points_raw, points1, K, distCoeffs);
    undistortPoints(next_points_raw, points2, K, distCoeffs);

    Matx34d prev_P, next_P;
    prev_P = Matx34d::eye();
    next_P = Matx34d(R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0),
            R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1),
            R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2));

    triangulatePoints(prev_P, next_P, points1, points2, XP);
    convertPointsFromHomogeneous(XP.t(), pt_3d);
    if (!TestTriangulation(pt_3d, next_P, status)) {
        next_P = Matx34d(R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t2.at<double>(0),
                R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t2.at<double>(1),
                R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t2.at<double>(2));

        triangulatePoints(prev_P, next_P, points1, points2, XP);
        cout << "Test triangulate out: 2" << endl;
        convertPointsFromHomogeneous(XP.t(), pt_3d);

        if (!TestTriangulation(pt_3d, next_P, status)) {
            next_P = Matx34d(R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t1.at<double>(0),
                    R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t1.at<double>(1),
                    R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t1.at<double>(2));

            triangulatePoints(prev_P, next_P, points1, points2, XP);
            cout << "Test triangulate out: 3" << endl;
            convertPointsFromHomogeneous(XP.t(), pt_3d);

            if (!TestTriangulation(pt_3d, next_P, status)) {

                next_P = Matx34d(R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0),
                        R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1),
                        R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2));
                triangulatePoints(prev_P, next_P, points1, points2, XP);
                cout << "Test triangulate out: 4" << endl;
                convertPointsFromHomogeneous(XP.t(), pt_3d);

                if (!TestTriangulation(pt_3d, next_P, status)) {
                    R = Mat::eye(3, 3, CV_64FC1);
                    T = Mat::zeros(3, 1, CV_64FC1);
                    cout << "Test triangulate out: 4 - not good estimation" << endl;
                } else {
                    R = R2;
                    T = t2;
                    cout << "Test triangulate out: 4 - proper" << endl;
                }
            }else {
                R = R2;
                T = t1;
            }
        } else {
            R = R1;
            T = t2;
        }
    } else {
        R = R1;
        T = t1;
    }
}

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> LinearLSTriangulation(Point3d u,		//homogenous image point (u,v,1)
        Matx34d P,		//camera 1 matrix
        Point3d u1,		//homogenous image point in 2nd camera
        Matx34d P1		//camera 2 matrix
)
{

    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    //	cout << "u " << u <<", u1 " << u1 << endl;
    //	Matx<double,6,4> A; //this is for the AX=0 case, and with linear dependence..
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u.x*P(1)-u.y*P(0);
    //	A(3) = u1.x*P1(2)-P1(0);
    //	A(4) = u1.y*P1(2)-P1(1);
    //	A(5) = u1.x*P(1)-u1.y*P1(0);
    //	Matx43d A; //not working for some reason...
    //	A(0) = u.x*P(2)-P(0);
    //	A(1) = u.y*P(2)-P(1);
    //	A(2) = u1.x*P1(2)-P1(0);
    //	A(3) = u1.y*P1(2)-P1(1);
    Matx43d A(u.x*P(2,0)-P(0,0),	u.x*P(2,1)-P(0,1),		u.x*P(2,2)-P(0,2),
            u.y*P(2,0)-P(1,0),	u.y*P(2,1)-P(1,1),		u.y*P(2,2)-P(1,2),
            u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),	u1.x*P1(2,2)-P1(0,2),
            u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),	u1.y*P1(2,2)-P1(1,2)
    );
    Matx41d B(-(u.x*P(2,3)	-P(0,3)),
            -(u.y*P(2,3)	-P(1,3)),
            -(u1.x*P1(2,3)	-P1(0,3)),
            -(u1.y*P1(2,3)	-P1(1,3)));

    Mat_<double> X;
    solve(A,B,X,DECOMP_SVD);

    return X;
}


/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> IterativeLinearLSTriangulation(Point3d u,	//homogenous image point (u,v,1)
        Matx34d P,			//camera 1 matrix
        Point3d u1,			//homogenous image point in 2nd camera
        Matx34d P1			//camera 2 matrix
) {
    double wi = 1, wi1 = 1;
    Mat_<double> X(4,1);
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;

        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

        //breaking point
        if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x*P(2,0)-P(0,0))/wi,		(u.x*P(2,1)-P(0,1))/wi,			(u.x*P(2,2)-P(0,2))/wi,
                (u.y*P(2,0)-P(1,0))/wi,		(u.y*P(2,1)-P(1,1))/wi,			(u.y*P(2,2)-P(1,2))/wi,
                (u1.x*P1(2,0)-P1(0,0))/wi1,	(u1.x*P1(2,1)-P1(0,1))/wi1,		(u1.x*P1(2,2)-P1(0,2))/wi1,
                (u1.y*P1(2,0)-P1(1,0))/wi1,	(u1.y*P1(2,1)-P1(1,1))/wi1,		(u1.y*P1(2,2)-P1(1,2))/wi1
        );
        Mat_<double> B = (Mat_<double>(4,1) <<	  -(u.x*P(2,3)	-P(0,3))/wi,
                -(u.y*P(2,3)	-P(1,3))/wi,
                -(u1.x*P1(2,3)	-P1(0,3))/wi1,
                -(u1.y*P1(2,3)	-P1(1,3))/wi1
        );

        solve(A,B,X_,DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X(3) = 1.0;
    }
    return X;
}