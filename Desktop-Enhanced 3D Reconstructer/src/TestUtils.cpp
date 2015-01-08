#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void prepareSynthData(){
    //    vector<Point3d> pointsSynth;
//    prepareSynthCube(pointsSynth);
//    Mat K1 = (Mat_<double>(3, 3) << K.at<double>(0), K.at<double>(1), K.at<double>(2), K.at<double>(3), K.at<double>(4), K.at<double>(5), K.at<double>(6), K.at<double>(7), K.at<double>(8));

//    Mat Rsynth1;
//    Mat Rsynth2;
//    Mat Tsynth1;
//    Mat Tsynth2;
//
//    Rsynth1 = getRotated3DMatrix<double>(0, 0, 0);
//    Rsynth2 = getRotated3DMatrix<double>(0, 10, 0);
//    Tsynth1 = (Mat_<double>(3, 1) << -10.0f, 0.0f, 0.0f);
//    Tsynth2 = (Mat_<double>(3, 1) << cos(radians(10)) * -10.0f, 0.0f, sin(radians(10)) * -10.0f);

//    vector<Point2d> points1synth;
//    vector<Point2d> points2synth;

//    projectPoints(pointsSynth, Rsynth1, Tsynth1, K1, distCoeffs, points1synth);
//    projectPoints(pointsSynth, Rsynth2, Tsynth2, K1, distCoeffs, points2synth);
}

double sampson_error(const double *f, double x1, double y1, double x2, double y2) {
    double Fx1[3] = {
            f[0] * x1 + f[1] * y1 + f[2],
            f[3] * x1 + f[4] * y1 + f[5],
            f[6] * x1 + f[7] * y1 + f[8]
    };
    double Ftx2[3] = {
            f[0] * x2 + f[3] * y2 + f[6],
            f[1] * x2 + f[4] * y2 + f[7],
            f[2] * x2 + f[5] * y2 + f[8]
    };
    double x2tFx1 = Fx1[0] * x2 + Fx1[1] * y2 + Fx1[2];

    double error = x2tFx1 * x2tFx1 / (Fx1[0] * Fx1[0] + Fx1[1] * Fx1[1] + Ftx2[0] * Ftx2[0] + Ftx2[1] * Ftx2[1]);
    error = sqrt(error);
    return error;

}

void prepareSynthCube(vector<Point3d> pointsSynth) {
    pointsSynth.push_back(Point3d(-3, -3, -3));
    pointsSynth.push_back(Point3d(-3, -3, 3));
    pointsSynth.push_back(Point3d(3, -3, -3));
    pointsSynth.push_back(Point3d(3, -3, 3));
    pointsSynth.push_back(Point3d(3, 3, -3));
    pointsSynth.push_back(Point3d(3, 3, 3));
    pointsSynth.push_back(Point3d(-3, 3, -3));
    pointsSynth.push_back(Point3d(-3, 3, 3));
}

bool TestTriangulation(const vector<Point3d> &pcloud_pt3d, const Matx34d &P, vector<uchar> &status) {
    vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());

    Matx44d P4x4 = Matx44d::eye();
    for (int i = 0; i < 12; i++) P4x4.val[i] = P.val[i];

    perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);

    status.resize(pcloud_pt3d.size(), 0);
    for (int i = 0; i < pcloud_pt3d.size(); i++) {
        status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
    }
    int count = countNonZero(status);

    double percentage = ((double) count / (double) pcloud_pt3d.size());
    cout << count << "/" << pcloud_pt3d.size() << " = " << percentage * 100.0 << "% are in front of camera" << endl;
    if (percentage < 0.75)
        return false; //less than 75% of the points are in front of the camera


    return true;
}

void interprete(Mat& Rot){

    Mat M = Rot.clone();

    cout << "M " << M <<endl;

    double lambda = sqrt(M.at<double>(2, 0)*M.at<double>(2, 0) + M.at<double>(2, 1)*M.at<double>(2, 1) + M.at<double>(2, 2)*M.at<double>(2, 2));
    double determinant = cv::determinant(M);
    if(determinant <= 0)
        lambda = - lambda;

    M = M / lambda;

    cout << "M normalized" << M <<endl;

    Mat Qx = Mat::eye(3,3,CV_64FC1);
    double t = sqrt(M.at<double>(2, 1)*M.at<double>(2, 1)+M.at<double>(2, 2)*M.at<double>(2, 2));
    double s = -M.at<double>(2, 1)/t;
    double c = M.at<double>(2, 2)/t;
    Qx.at<double>(1,1) = c;
    Qx.at<double>(2,2) = c;
    Qx.at<double>(1,2) = -s;
    Qx.at<double>(2,1) = s;

    Mat M1 = M * Qx;
    M1.at<double>(2, 1) = 0;

//    cout << "M1 " << M1 <<endl;

    Mat Qy = Mat::eye(3,3,CV_64FC1);
    t = sqrt(M1.at<double>(2, 0)*M1.at<double>(2, 0)+M1.at<double>(2, 2)*M1.at<double>(2, 2));
    s = M1.at<double>(2, 0)/t;
    c = M1.at<double>(2, 2)/t;
    Qy.at<double>(0,0) = c;
    Qy.at<double>(2,2) = c;
    Qy.at<double>(0,2) = s;
    Qy.at<double>(2,0) = -s;

    Mat M2 = M1 * Qy;
    M2.at<double>(2, 1) = 0;
    M2.at<double>(2, 0) = 0;
//    cout << "M2 " << M2 <<endl;

    Mat Qz = Mat::eye(3,3,CV_64FC1);
    t = sqrt(M2.at<double>(1, 0)*M2.at<double>(1, 0)+M2.at<double>(1, 1)*M2.at<double>(1, 1));
    s = -M2.at<double>(1, 0)/t;
    c = M2.at<double>(1, 1)/t;
    Qz.at<double>(0,0) = c;
    Qz.at<double>(1,1) = c;
    Qz.at<double>(0,1) = -s;
    Qz.at<double>(1,0) = s;

    Mat M3 = M2 * Qz;
    M3.at<double>(2, 1) = 0;
    M3.at<double>(2, 0) = 0;
    M3.at<double>(1, 0) = 0;
//    cout << "M3 " << M3 <<endl;

//    Mat R = M * Qx * Qy * Qz;
    Mat R = M3;

    Mat Q = Qz.t() * Qy.t() * Qx.t();

    double alphax = R.at<double>(0,0);
    double alphay = R.at<double>(1,1);
    double skew =   R.at<double>(0,1);

    double x0 = R.at<double>(0,2);
    double y0 = R.at<double>(1,2);
    double aspectRatio = alphay/alphax;

//    Mat C = -M.inv() * (P.col(3)/lambda);


    double omega = atan(-Q.at<double>(2,1)/Q.at<double>(2,2)) * 180 / 3.1415;
    double phi = asin(Q.at<double>(2,0)) * 180 / 3.1415;
    double kappa = atan(-Q.at<double>(1,0)/Q.at<double>(0,0)) * 180 / 3.1415;

    cout << "R - rotation matrix " << Q <<endl;
//    cout << "Projection center: " << C << endl;
    cout << "Rotation angles: omega - " << omega << " , phi - "<< phi << " , kappa - " << kappa << endl;

    cout << "K - calibration matrix " << R <<endl;
    cout << "Principle distance - Alpha x: " << alphax << endl;
    cout << "Alpha y: " << alphay << endl;
    cout << "Shearing - s " << skew << endl;
    cout << "Skew angle " << acos(-skew/alphax) * 180 / 3.1415 << endl;
    cout << "Principle point x0,y0: " << x0 << " , " << y0 << endl;
    cout << "Aspect ratio: " << aspectRatio << endl;


}