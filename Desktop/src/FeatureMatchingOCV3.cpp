#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

struct SURFDetector {
    Ptr<Feature2D> surf;

    SURFDetector(int numberFeatures) {
        surf = SIFT::create(numberFeatures);
    }

    template<class T>
    void operator()(const T &in, const T &mask, std::vector<cv::KeyPoint> &pts, T &descriptors, bool useProvided = false) {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};

void getSiftKeypointsAndDescriptors(Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors, int nFeatures) {
    SURFDetector surf(nFeatures);

//    DynamicAdaptedFeatureDetector surf(new FastAdjuster(10,true), 5000, 10000, 10);
    surf(image, Mat(), keypoints, descriptors);
}

void getCorrespondingPoints(vector<KeyPoint> &keypoints, vector<KeyPoint> &keypoints1, Mat &descriptors, Mat &descriptors1, vector<DMatch> &good_matches, vector<Point2d> &image1Points, vector<Point2d> &image2Points) {

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    SURFMatcher<FlannBasedMatcher> matcher;
    vector<DMatch>  matches;
    matcher.match(descriptors, descriptors1, matches);

    for (int i = 0; i < descriptors.rows; i++) {
        const float ratio = 0.8; // As in Lowe's paper; can be tuned
        good_matches.push_back(matches[i]);
    }

    for (int k = 0; k < good_matches.size(); k++) {
        image1Points.push_back(keypoints[good_matches[k].queryIdx].pt);
        image2Points.push_back(keypoints1[good_matches[k].trainIdx].pt);
    }
}