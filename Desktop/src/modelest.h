#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

int cvRANSACUpdateNumItersy( double p, double ep,
        int model_points, int max_iters );
int findInliersy( const vector<Point2d> &m1, const vector<Point2d> &m2,
        Mat F, float* err,
        uchar* mask, double threshold );
void computeReprojErrory( const vector<Point2d> &m1, const vector<Point2d> &m2,
        Mat F, float* err);
bool checkSubsety( const vector<Point2d> &m, int count );
bool getSubsety( const vector<Point2d> &m1, const vector<Point2d> &m2,
        vector<Point2d> &ms1, vector<Point2d> &ms2, int maxAttempts, int modelPoints );