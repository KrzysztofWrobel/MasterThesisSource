#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;


double radians(double d) {
    return d * M_PI / 180;
}

double degrees(double r) {
    return r * 180 / M_PI;
}

void cv::convertPointsToHomogeneous(const InputArray &_src, OutputArray _dst) {
    Mat src = _src.getMat();
    int npoints = src.checkVector(2), cn = 2;
    if (npoints < 0) {
        npoints = src.checkVector(3);
        if (npoints >= 0)
            cn = 3;
    }
    CV_Assert(npoints >= 0 && (src.depth() == CV_64F));

    _dst.create(npoints, 1, CV_MAKETYPE(CV_64F, cn + 1));
    CvMat c_src = src, c_dst = _dst.getMat();
    cvConvertPointsHomogeneous(&c_src, &c_dst);
}

void cv::convertPointsFromHomogeneous(const InputArray &_src, OutputArray _dst) {
    Mat src = _src.getMat();
    int npoints = src.checkVector(3), cn = 3;
    if (npoints < 0) {
        npoints = src.checkVector(4);
        if (npoints >= 0)
            cn = 4;
    }
    CV_Assert(npoints >= 0 && (src.depth() == CV_64F));

    _dst.create(npoints, 1, CV_MAKETYPE(CV_64F, cn - 1));
    CvMat c_src = src, c_dst = _dst.getMat();
    cvConvertPointsHomogeneous(&c_src, &c_dst);
}

Mat getRotated4DMatrix(double pitch, double azimuth, double roll);

template<typename T>
Mat getRotated3DMatrix(double pitch, double azimuth, double roll);

template<typename T>
Mat getRotation3DMatrix(double pitch, double azimuth, double roll);

void computeCorrespondEpilines64(const InputArray &_points, int whichImage,
        const InputArray &_Fmat, OutputArray _lines) {
    Mat points = _points.getMat(), F = _Fmat.getMat();
    int npoints = points.checkVector(2);
    CV_Assert(npoints >= 0 && (points.depth() == CV_64F));

    _lines.create(npoints, 1, CV_64FC3, -1, true);

    CvMat c_points = points, c_lines = _lines.getMat(), c_F = F;
    cvComputeCorrespondEpilines(&c_points, whichImage, &c_F, &c_lines);
}

template<typename T>
static float distancePointLine(const cv::Point_<T> point, const cv::Vec<T, 3> &line) {
    //Line is given as a*x + b*y + c = 0
    return std::fabsf(line(0) * point.x + line(1) * point.y + line(2))
            / std::sqrt(line(0) * line(0) + line(1) * line(1));
}

template<typename T1, typename T2>
static void drawEpipolarLines(const std::string &title, const cv::Matx<T1, 3, 3> F, const cv::Mat &img1, const cv::Mat &img2,
        const std::vector<cv::Point_<T2> > points1,
        const std::vector<cv::Point_<T2> > points2,
        Mat& outIMG,
        const cv::Rect& whereRegion,
        const float inlierDistance = -1) {
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());
    cv::Mat outImg(img1.rows, img1.cols * 2, CV_8UC3);
    cv::Rect rect1(0, 0, img1.cols, img1.rows);
    cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
    /*
     * Allow color drawing
     */
    if (img1.type() == CV_8U) {
        cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
        cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
    }
    else {
        img1.copyTo(outImg(rect1));
        img2.copyTo(outImg(rect2));
    }
    std::vector<cv::Vec<T2, 3> > epilines1, epilines2;
    computeCorrespondEpilines64(points1, 1, F, epilines1); //Index starts with 1
    computeCorrespondEpilines64(points2, 2, F, epilines2);

    CV_Assert(points1.size() == points2.size() &&
            points2.size() == epilines1.size() &&
            epilines1.size() == epilines2.size());

    cv::RNG rng(0);
    for (size_t i = 0; i < points1.size(); i++) {
        if (inlierDistance > 0) {
            if (distancePointLine<T2>(points1[i], epilines2[i]) > inlierDistance ||
                    distancePointLine<T2>(points2[i], epilines1[i]) > inlierDistance) {
                //The point match is no inlier
                continue;
            }
        }
        /*
         * Epipolar lines of the 1st point set are drawn in the 2nd image and vice-versa
         */
        cv::Scalar color(rng(256), rng(256), rng(256));

        Mat outImg1 = outImg(rect1);
        Mat outImg2 = outImg(rect2);
        cv::line(outImg2,
                cv::Point(0, -epilines1[i][2] / epilines1[i][1]),
                cv::Point(img1.cols, -(epilines1[i][2] + epilines1[i][0] * img1.cols) / epilines1[i][1]),
                color);
        cv::circle(outImg1, points1[i], 3, color, -1, CV_AA);

        cv::line(outImg1,
                cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
                cv::Point(img2.cols, -(epilines2[i][2] + epilines2[i][0] * img2.cols) / epilines2[i][1]),
                color);
        cv::circle(outImg2, points2[i], 3, color, -1, CV_AA);
    }
    cv::imshow(title, outImg);
    if (outIMG.type() == CV_8U) {
        cv::cvtColor(outImg, outIMG(whereRegion), CV_GRAY2BGR);
    } else {
        outImg.copyTo(outIMG(whereRegion));
    }
    cv::waitKey(1);
}

void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f) {
    // get width and height for ease of use in matrices
    double w = (double) input.cols;
    double h = (double) input.rows;
    // Projection 2D -> 3D matrix
    Mat A1 = (Mat_<double>(4, 3) <<
            1, 0, -w / 2,
            0, 1, -h / 2,
            0, 0, 0,
            0, 0, 1);
    // Rotation matrices around the X, Y, and Z axis
    Mat RX = (Mat_<double>(4, 4) <<
            1, 0, 0, 0,
            0, cos(alpha), sin(alpha), 0,
            0, -sin(alpha), cos(alpha), 0,
            0, 0, 0, 1);
    Mat RY = (Mat_<double>(4, 4) <<
            cos(beta), 0, -sin(beta), 0,
            0, 1, 0, 0,
            sin(beta), 0, cos(beta), 0,
            0, 0, 0, 1);
    Mat RZ = (Mat_<double>(4, 4) <<
            cos(gamma), sin(gamma), 0, 0,
            -sin(gamma), cos(gamma), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    // Composed rotation matrix with (RX, RY, RZ)
    Mat R = RX * RY * RZ;
    // Translation matrix
    Mat T = (Mat_<double>(4, 4) <<
            1, 0, 0, dx,
            0, 1, 0, dy,
            0, 0, 1, dz,
            0, 0, 0, 1);
    // 3D -> 2D matrix
    Mat A2 = (Mat_<double>(3, 4) <<
            f, 0, w / 2, 0,
            0, f, h / 2, 0,
            0, 0, 1, 0);
    // Final transformation matrix
    Mat trans = (A2 * (T * (R * (A1))));
    // Apply matrix transformation
    warpPerspective(input, output, trans, input.size(), INTER_LANCZOS4);
}

Mat getRotated4DMatrix(double pitch, double azimuth, double roll) {
    Mat RX = (Mat_<double>(4, 4) <<
            1, 0, 0, 0,
            0, cos(pitch), -sin(pitch), 0,
            0, sin(pitch), cos(pitch), 0,
            0, 0, 0, 1);

    Mat RY = (Mat_<double>(4, 4) <<
            cos(azimuth), 0, -sin(azimuth), 0,
            0, 1, 0, 0,
            sin(azimuth), 0, cos(azimuth), 0,
            0, 0, 0, 1);
    Mat RZ = (Mat_<double>(4, 4) <<
            cos(roll), -sin(roll), 0, 0,
            sin(roll), cos(roll), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    //Important
    return RY * RZ * RX;
}

template<typename T>
Mat getRotated3DMatrix(double pitch, double azimuth, double roll) {
    Mat RX = (Mat_<T>(3, 3) <<
            1, 0, 0,
            0, cos(pitch), sin(pitch),
            0, -sin(pitch), cos(pitch));

    Mat RY = (Mat_<T>(3, 3) <<
            cos(azimuth), 0, -sin(azimuth),
            0, 1, 0,
            sin(azimuth), 0, cos(azimuth));
    Mat RZ = (Mat_<T>(3, 3) <<
            cos(roll), sin(roll), 0,
            -sin(roll), cos(roll), 0,
            0, 0, 1);
    //Important
    return RZ * RY * RX;
}

//http://mathworld.wolfram.com/EulerAngles.html
template<typename T>
Mat getRotation3DMatrix(double pitch, double azimuth, double roll) {
    Mat D = (Mat_<T>(3, 3) <<
            cos(roll), -sin(roll), 0,
            sin(roll), cos(roll), 0,
            0, 0, 1);

    Mat C = (Mat_<T>(3, 3) <<
            cos(azimuth), 0, -sin(azimuth),
            0, 1, 0,
            sin(azimuth), 0, cos(azimuth));
    Mat B = (Mat_<T>(3, 3) <<
            1, 0, 0,
            0, cos(pitch), -sin(pitch),
            0, sin(pitch), cos(pitch));
    //Important
    return B * C * D;
}

