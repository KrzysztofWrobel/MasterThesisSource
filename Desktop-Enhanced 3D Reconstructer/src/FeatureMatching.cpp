#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void getSiftKeypointsAndDescriptors(Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors, int nFeatures) {
    SiftFeatureDetector surf(nFeatures);
    SiftDescriptorExtractor extractor;

//    DynamicAdaptedFeatureDetector surf(new FastAdjuster(10,true), 5000, 10000, 10);
    surf.detect(image, keypoints);
    extractor.compute(image, keypoints, descriptors);
}

void getCorrespondingPoints(vector<KeyPoint> &keypoints, vector<KeyPoint> &keypoints1, Mat &descriptors, Mat &descriptors1, vector<DMatch> &good_matches, vector<Point2d> &image1Points, vector<Point2d> &image2Points) {
    good_matches.clear();

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    vector<vector<DMatch> > matches;
    matcher.knnMatch(descriptors, descriptors1, matches, 2);

    for (int i = 0; i < descriptors.rows; i++) {
        const float ratio = 0.8; // As in Lowe's paper; can be tuned
        if (matches[i][0].distance < ratio * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }

    for (int k = 0; k < good_matches.size(); k++) {
        image1Points.push_back(keypoints[good_matches[k].queryIdx].pt);
        image2Points.push_back(keypoints1[good_matches[k].trainIdx].pt);
    };
}