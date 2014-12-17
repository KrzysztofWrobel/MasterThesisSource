#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void prepareSynthData();
void prepareSynthCube(vector<Point3d> pointsSynth);

double sampson_error(const double *f, double x1, double y1, double x2, double y2);

bool TestTriangulation(const vector<Point3d> &pcloud_pt3d, const Matx34d &P, vector<uchar> &status);

void interprete(Mat& Rot);