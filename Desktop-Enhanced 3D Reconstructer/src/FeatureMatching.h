
#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void getSiftKeypointsAndDescriptors(Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors, int nFeatures);
void getCorrespondingPoints(vector<KeyPoint> &keypoints, vector<KeyPoint> &keypoints1, Mat &descriptors, Mat &descriptors1, vector<DMatch> &good_matches, vector<Point2d> &image1Points, vector<Point2d> &image2Points);