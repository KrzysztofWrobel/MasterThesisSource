/* Example 10-1. Pyramid Lucas-Kanade optical flow code
 
 ************************************************** */
#include <iostream>
#include <fstream>
#include <cassert>
#include <exception>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/nonfree/nonfree.hpp>

#ifdef _MSC_VER
#include <boost/config/compiler/visualc.hpp>
#endif
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <stdio.h>
#include <time.h>

#define MAX_CORNERS 100

using namespace std;
using namespace cv;

Mat gray;
Mat K, distCoeffs;
vector<Point2f> grabbedPoints, prevPoints, nextPoints, u_grabbedPoints, u_nextPoints;

void defineCameras(Mat& F, Mat& P1, Mat& P2);
Mat linearTriangulation(Mat& P1, Mat& P2, vector<Point2f>& x1, vector<Point2f>& x2);
void savePointList(string fname, Mat& points);
bool TestTriangulation(const vector<Point3f>& pcloud_pt3d, const Matx34d& P, vector<uchar>& status) ;
bool FindPoseEstimation(
                        int working_view,
                        cv::Mat_<double>& rvec,
                        cv::Mat_<double>& t,
                        cv::Mat_<double>& R,
                        std::vector<cv::Point3f> ppcloud,
                        std::vector<cv::Point2f> imgPoints
                        );

int main(int argc, char** argv) {

	/*********************************************************************************************
     First we get the video
     *********************************************************************************************/
	VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    cv::FileStorage fs;
    fs.open("out_camera_data.yml",cv::FileStorage::READ);
    fs["camera_matrix"]>>K;
    fs["distortion_coefficients"]>>distCoeffs;
    K.convertTo(K, CV_32FC1);
    cout << K << endl;

    try
    {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json("sensor.txt", pt);

        cout << pt.size() << endl;

        BOOST_FOREACH(boost::property_tree::ptree::value_type &v, pt)
                    {
                        assert(v.first.empty()); // array elements have no names
                        std::cout << v.first.data() << std::endl;
                        std::string path = v.second.get<std::string>("photoPath","none");
                        std::cout << path << std::endl;
                        // etc
                    }
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << std::endl;
    }
    
    Mat frame, prev_frame;
    std::vector<uchar> status, trackedStatus;
    std::vector<float> err;
    cap >> frame; // get a new frame from camera
    
    cvtColor(frame, gray, CV_RGB2GRAY);
    GaussianBlur(gray, gray, Size(5,5), 1.5, 1.5);
//    Canny(gray, gray, 0, 30, 3);
    
    vector<KeyPoint> v;

    
    // extract keypoints from the first image
//    SURF surf_extractor(5.0e3);
//        surf_extractor(gray, Mat(), v);
//    vector<KeyPoint> keypoints1;
    
    // printf("Extracting keypoints\n");

    
    SiftFeatureDetector surf(1200);
    surf.detect(gray, v);
//    FastFeatureDetector detector(50);
//    detector.detect(gray, v);
//    Sift
//        detector.detect(gray, v);
//
//    goodFeaturesToTrack(
//                        gray,					//Input 8-bit or floating-point 32-bit, single-channel image.
//                        prevPoints,				//The parameter is ignored. Same size floating-point 32-bit, single-channel image.
//                        MAX_CORNERS,			//Maximum number of corners to return. It is overwritten by the number of points returned
//                        0.01,					//Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
//                        3.0					//Minimum possible Euclidean distance between the returned corners.
//                        );
    
//    KeyPoint::convert(v, prevPoints);
//    Mat mPrevPoint = Mat(prevPoints);
//    mPrevPoint.convertTo(mPrevPoint,CV_32FC2);
//    prevPoints = (vector<Point2f>)mPrevPoint;
    
    prev_frame = gray.clone();
    grabbedPoints = prevPoints;
    Matx34d lastCamera;
    
    Mat XP = Mat::zeros(4,1,CV_32FC1);
    
    bool init = false;

    int cnt = 0;
    int a = 0;
    namedWindow("edges",1);
    
    for(;;)
    {
        cap >> frame; // get a new frame from camera
        cvtColor(frame, gray, CV_RGB2GRAY);
        GaussianBlur(gray, gray, Size(7,7), 1.5, 1.5);
//        Canny(gray, gray, 0, 30, 3);
        
        
        if(cnt > 20){
            if(!init){
                init = true;
            } else {
            cnt = 0;
            a++;
                vector<Point2f> goodGrabbed, goodNext;
                for (unsigned int i=0; i<trackedStatus.size(); i++) {
                    if (trackedStatus[i])
                    {
                        goodGrabbed.push_back(grabbedPoints[i]);
                        goodNext.push_back(nextPoints[i]);
                    }
                }
                
                if(countNonZero(trackedStatus) < 8){
                    continue;
                }
                
            grabbedPoints = goodGrabbed;
            nextPoints = goodNext;

                
            Mat F = findFundamentalMat(grabbedPoints, nextPoints, FM_RANSAC, 0.01, 0.99, status);
            vector<DMatch> new_matches;
            cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;
                
                
            F.convertTo(F, CV_32FC1);
            Mat E = K.t() * F * K;
            cout << "F: " << F;
            
            Matx34d P1,P2,P2a,P2b,P2c;
                Mat XP1, XP2, XP3;
            
            P1 = Matx34d::eye();
            
            SVD svd(E);
            Mat W = Mat(Matx33d (0,-1,0,   //HZ 9.13
                      1,0,0,
                      0,0,1));
                W.convertTo(W, CV_32FC1);
            Mat Winv = Mat(Matx33d(0,1,0,
                         -1,0,0,
                         0,0,1));
                Winv.convertTo(Winv, CV_32FC1);
            Mat R = svd.u * W * svd.vt; //HZ 9.19
            Mat R1 = svd.u * Winv * svd.vt; //HZ 9.19
            Mat t = svd.u.col(2); //u3
                Mat t1 = -svd.u.col(2);
            P2 = Matx34d(R.at<float>(0,0),    R.at<float>(0,1), R.at<float>(0,2), t.at<float>(0),
                         R.at<float>(1,0),    R.at<float>(1,1), R.at<float>(1,2), t.at<float>(1),
                         R.at<float>(2,0),    R.at<float>(2,1), R.at<float>(2,2), t.at<float>(2));

                
            P2a = Matx34d(R.at<float>(0,0),    R.at<float>(0,1), R.at<float>(0,2), t1.at<float>(0),
                                 R.at<float>(1,0),    R.at<float>(1,1), R.at<float>(1,2), t1.at<float>(1),
                                 R.at<float>(2,0),    R.at<float>(2,1), R.at<float>(2,2), t1.at<float>(2));
                
            P2b = Matx34d(R1.at<float>(0,0),    R1.at<float>(0,1), R1.at<float>(0,2), t.at<float>(0),
                                 R1.at<float>(1,0),    R1.at<float>(1,1), R1.at<float>(1,2), t.at<float>(1),
                                 R1.at<float>(2,0),    R1.at<float>(2,1), R1.at<float>(2,2), t.at<float>(2));
                
            P2c = Matx34d(R1.at<float>(0,0),    R1.at<float>(0,1), R1.at<float>(0,2), t1.at<float>(0),
                                 R1.at<float>(1,0),    R1.at<float>(1,1), R1.at<float>(1,2), t1.at<float>(1),
                                 R1.at<float>(2,0),    R1.at<float>(2,1), R1.at<float>(2,2), t1.at<float>(2));
                

            vector<Point3f> x1h,x2h;
                
                Mat pt_set1_pt,pt_set2_pt;
                undistortPoints(grabbedPoints, u_grabbedPoints, K, distCoeffs);
                undistortPoints(nextPoints, u_nextPoints, K, distCoeffs);
                
                int max;
                
                triangulatePoints(P1, P2, u_grabbedPoints, u_nextPoints, XP);
                //calculate reprojection
                vector<Point3f> pt_3d;
                convertPointsHomogeneous(XP.reshape(4, 1), pt_3d);
                status.clear();
                if(!TestTriangulation(pt_3d, P2, status)){
                    max = countNonZero(status);
                    
                    triangulatePoints(P1, P2a, u_grabbedPoints, u_nextPoints, XP1);
                    cout<< "Test triangulate out: 2"<< endl;
                    vector<Point3f> pt_3d;
                    convertPointsHomogeneous(XP1.reshape(4, 1), pt_3d);
                    
                    if(!TestTriangulation(pt_3d, P2a, status)){
                        if(countNonZero(status) > max){
                            XP = XP1;
                            P2 = P2a;
                        }
                        triangulatePoints(P1, P2b, u_grabbedPoints, u_nextPoints, XP2);
                        cout<< "Test triangulate out: 3"<< endl;
                        vector<Point3f> pt_3d;
                        convertPointsHomogeneous(XP2.reshape(4, 1), pt_3d);
                    
                        if(!TestTriangulation(pt_3d, P2b, status)){
                            if(countNonZero(status) > max){
                                XP = XP2;
                                P2 = P2b;
                            }
                            triangulatePoints(P1, P2c, u_grabbedPoints, u_nextPoints, XP3);
                            cout<< "Test triangulate out: 4"<< endl;
                            vector<Point3f> pt_3d;
                            convertPointsHomogeneous(XP3.reshape(4, 1), pt_3d);
                        
                            if(!TestTriangulation(pt_3d, P2c, status)){
                                if(countNonZero(status) > max){
                                    XP = XP3;
                                    P2 = P2c;
                                }
                                
                            } else {
                                cout<< "Test triangulate out: 4 - proper"<< endl;
                                XP = XP3;
                                P2 = P2c;
                            }
                        } else {
                            XP = XP2;
                            P2 = P2b;
                        }
                    } else {
                        XP = XP1;
                        P2 = P2a;
                    }
                } else{
                    cout<< "Test triangulate out: 1"<< endl;
                }
            
            
            cout << endl << "Camera 1: " << P1 << endl;
            cout << endl << "Camera 2: " << P2 << endl;
            
            lastCamera = P2;

//            cout << "x2: " << x << endl;
//            cout << "Reconstructed control points: " << XP << endl;
            

            savePointList("projectiveReconstruction.asc", XP);
                cv::Mat_<double> tr = (cv::Mat_<double>(1,3) << P2(0,3), P2(1,3), P2(2,3));
                cv::Mat_<double> Ro = (cv::Mat_<double>(3,3) << P2(0,0), P2(0,1), P2(0,2),
                                      P2(1,0), P2(1,1), P2(1,2),
                                      P2(2,0), P2(2,1), P2(2,2));
                cv::Mat_<double> rvec(1,3); Rodrigues(Ro, rvec);
                
                vector<Point3f> xp_p;
                convertPointsHomogeneous(XP.reshape(4, 1), xp_p);
                bool found = FindPoseEstimation(2, rvec, tr, Ro, xp_p, u_nextPoints);

            }
//            SURF surf_extractor(5.0e3);
//            surf_extractor(gray, Mat(), v);
            SiftFeatureDetector surf(800);
            surf.detect(gray, v);
            
//            SurfDescriptorExtractor extractor;
            
//            Mat descriptors_1, descriptors_2;
            
//            extractor.compute( gray, v, descriptors_1 );
//            extractor.compute( img_2, keypoints_2, descriptors_2 );
//            FastFeatureDetector detector(20);
//            detector.detect(gray, v);
            //
            //    goodFeaturesToTrack(
            //                        gray,					//Input 8-bit or floating-point 32-bit, single-channel image.
            //                        prevPoints,				//The parameter is ignored. Same size floating-point 32-bit, single-channel image.
            //                        MAX_CORNERS,			//Maximum number of corners to return. It is overwritten by the number of points returned
            //                        0.01,					//Parameter characterizing the minimal accepted quality of image corners. The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue (see cornerMinEigenVal() ) or the Harris function response (see cornerHarris() ). The corners with the quality measure less than the product are rejected. For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected.
            //                        3.0					//Minimum possible Euclidean distance between the returned corners.
            //                        );
            
            KeyPoint::convert(v, prevPoints);
            Mat mPrevPoint = Mat(prevPoints);
            mPrevPoint.convertTo(mPrevPoint,CV_32FC2);
            prevPoints = (vector<Point2f>)mPrevPoint;
            grabbedPoints = prevPoints;
            
        } else {
            cnt++;

        }
        
        if(init){
            calcOpticalFlowPyrLK(
                             prev_frame, gray, // 2 consecutive images
                             prevPoints, // input point positions in first im
                             nextPoints, // output point positions in the 2nd
                             trackedStatus,    // tracking success
                             err      // tracking error
                             );
            int count = countNonZero(trackedStatus);
            
            cout << "Still tracked: " << count << endl;
            
        
            for( unsigned int i = 0; i < prevPoints.size(); i++ )
            {
                circle(frame, nextPoints[i], 10, Scalar(255,0,0,255));
                line(frame, grabbedPoints[i], nextPoints[i], Scalar(255,255,255,255));
//                cout << "Status: " << err[i];
            }
            



            imshow("edges", frame);
            prev_frame = gray.clone();
        
            prevPoints = nextPoints;
        }

        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

// calculates epipols from fundamental matrix
/*
 F	the fundamental matrix
 e1	first epipol
 e2	second epipol
 */
void getEpipols(Mat& F, Mat& e1, Mat& e2){
    
    SVD svd(F,SVD::FULL_UV);
    
    //    cout << "SVD:" << endl;
    //    cout << svd.u << endl;
    //    cout << svd.w << endl;
    //    cout << svd.vt << endl;
    
    float minVal = svd.w.at<float>(0,0);
    int minIdx = 0;
    
    for(int i = 1;i<svd.w.rows;i++){
        if(minVal > svd.w.at<float>(0,i)){
            minVal = svd.w.at<float>(0,i);
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
Mat makeSkewMatrix(Mat& e){
    Mat sk = Mat::zeros(3,3,CV_32FC1);
    sk.at<float>(0, 1) = -e.at<float>(2, 0);
    sk.at<float>(0, 2) = e.at<float>(1, 0);
    sk.at<float>(1, 0) = e.at<float>(2, 0);
    sk.at<float>(1, 2) = -e.at<float>(0, 0);
    sk.at<float>(2, 0) = -e.at<float>(1, 0);
    sk.at<float>(2, 1) = e.at<float>(0, 0);
    
    return sk;
}


void defineCameras(Mat& F, Mat& P1, Mat& P2){
    
    P1 = Mat::eye(3, 4, CV_32FC1);
    
    Mat e1;
    Mat e2;
    getEpipols(F, e1, e2);
    cout << "e1 = " << e1 <<endl;
    cout << "e2 = " << e2 <<endl;
    Mat sk = makeSkewMatrix(e2);
    cout << "sk(e2) = " << sk <<endl;
    
    P2 = sk*F;
    cv::hconcat(P2, e2, P2);
}

// solve homogeneous equation system by usage of SVD
/*
 A		the design matrix
 return		the estimated fundamental matrix
 */
Mat solve_dlt(Mat& A){
    
    SVD svd(A,SVD::FULL_UV);
    
    //    cout << "SVD:" << endl;
    //    cout << svd.u << endl;
    //    cout << svd.vt << endl;
    
    float minVal = svd.w.at<float>(0,0);
    int minIdx = 0;
    
    for(int i = 1;i<svd.w.rows;i++){
        if(minVal > svd.w.at<float>(0,i)){
            minVal = svd.w.at<float>(0,i);
            minIdx = i;
        }
    }
    
    //    cout << minVal << endl;
    //    cout << minIdx << endl;
    
    return svd.vt.row(minIdx).clone();
}

Mat linearTriangulation(Mat& P1, Mat& P2, vector<Point2f>& x1, vector<Point2f>& x2){
    
    Mat XP = Mat(0,4,CV_32FC1);
    
    for(int i=0;i<x1.size();i++){
        
        Mat p11 = x1[i].x * P1.row(2) - P1.row(0);
        Mat p12 = x1[i].y * P1.row(2) - P1.row(1);
        Mat p21 = x2[i].x * P2.row(2) - P2.row(0);
        Mat p22 = x2[i].y * P2.row(2) - P2.row(1);
        
        Mat A = Mat::zeros(0,4,CV_32FC1);
        
        A.push_back(p11);
        A.push_back(p12);
        A.push_back(p21);
        A.push_back(p22);
        
        Mat xp = solve_dlt(A);
        XP.push_back(xp);
    }
    
    return XP.t();
}

/* ***********************
 *** Given Functions ***
 *********************** */

// saves point list to file
/*
 fname		file name
 points		matrix of points
 */
void savePointList(string fname, Mat& points){
    
    // open file for write
    fstream file(fname.c_str(), ios::out);
    if(!file){
        cerr << "ERROR: cannot open file " << fname << endl;
        return;
    }
    
    // if homogeneous points: norm and write points
    if (points.rows == 4){
        for(int i=0; i<points.cols; i++){
            file << points.at<float>(0, i)/points.at<float>(3, i) << "," << points.at<float>(1, i)/points.at<float>(3, i) << "," << points.at<float>(2, i)/points.at<float>(3, i) << endl;
        }
    }
    // if euclidian points: write points
    if (points.rows == 3){
        for(int i=0; i<points.cols; i++){
            file << points.at<float>(0, i) << "," << points.at<float>(1, i) << "," << points.at<float>(2, i) << endl;
        }
    }
    
    // close file
    file.close();
}

bool FindPoseEstimation(
                                        int working_view,
                                        cv::Mat_<double>& rvec,
                                        cv::Mat_<double>& t,
                                        cv::Mat_<double>& R,
                                        std::vector<cv::Point3f> ppcloud,
                                        std::vector<cv::Point2f> imgPoints
                                        )
{
	if(ppcloud.size() <= 7 || imgPoints.size() <= 7 || ppcloud.size() != imgPoints.size()) {
		//something went wrong aligning 3D to 2D points..
		cerr << "couldn't find [enough] corresponding cloud points... (only " << ppcloud.size() << ")" <<endl;
		return false;
	}
    
	vector<int> inliers;

    //use CPU
    double minVal,maxVal; cv::minMaxIdx(imgPoints,&minVal,&maxVal);
    cv::solvePnPRansac(ppcloud, imgPoints, K, distCoeffs, rvec, t, true, 1000, 0.006 * maxVal, 0.25 * (double)(imgPoints.size()), inliers, CV_EPNP);
    
	vector<cv::Point2f> projected3D;
	cv::projectPoints(ppcloud, rvec, t, K, distCoeffs, projected3D);
    
	if(inliers.size()==0) { //get inliers
		for(int i=0;i<projected3D.size();i++) {
			if(norm(projected3D[i]-imgPoints[i]) < 10.0)
				inliers.push_back(i);
		}
	}
    
#if 0
	//display reprojected points and matches
	cv::Mat reprojected; imgs_orig[working_view].copyTo(reprojected);
	for(int ppt=0;ppt<imgPoints.size();ppt++) {
		cv::line(reprojected,imgPoints[ppt],projected3D[ppt],cv::Scalar(0,0,255),1);
	}
	for (int ppt=0; ppt<inliers.size(); ppt++) {
		cv::line(reprojected,imgPoints[inliers[ppt]],projected3D[inliers[ppt]],cv::Scalar(0,0,255),1);
	}
	for(int ppt=0;ppt<imgPoints.size();ppt++) {
		cv::circle(reprojected, imgPoints[ppt], 2, cv::Scalar(255,0,0), CV_FILLED);
		cv::circle(reprojected, projected3D[ppt], 2, cv::Scalar(0,255,0), CV_FILLED);
	}
	for (int ppt=0; ppt<inliers.size(); ppt++) {
		cv::circle(reprojected, imgPoints[inliers[ppt]], 2, cv::Scalar(255,255,0), CV_FILLED);
	}
	stringstream ss; ss << "inliers " << inliers.size() << " / " << projected3D.size();
	putText(reprojected, ss.str(), cv::Point(5,20), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,255), 2);
    
	cv::imshow("__tmp", reprojected);
	cv::waitKey(0);
	cv::destroyWindow("__tmp");
#endif
	//cv::Rodrigues(rvec, R);
	//visualizerShowCamera(R,t,0,255,0,0.1);
    
	if(inliers.size() < (double)(imgPoints.size())/5.0) {
		cerr << "not enough inliers to consider a good pose ("<<inliers.size()<<"/"<<imgPoints.size()<<")"<< endl;
		return false;
	}
    
	if(cv::norm(t) > 200.0) {
		// this is bad...
		cerr << "estimated camera movement is too big, skip this camera\r\n";
		return false;
	}
    
	cv::Rodrigues(rvec, R);
//	if(!CheckCoherentRotation(R)) {
//		cerr << "rotation is incoherent. we should try a different base view..." << endl;
//		return false;
//	}
    
	std::cout << "found t = " << t << "\nR = \n"<<R<<std::endl;
    
    return true;
}

bool TestTriangulation(const vector<Point3f>& pcloud_pt3d, const Matx34d& P, vector<uchar>& status) {
	vector<Point3f> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye();
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud_pt3d.size(),0);
	for (int i=0; i<pcloud_pt3d.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);
    
	double percentage = ((double)count / (double)pcloud_pt3d.size());
	cout << count << "/" << pcloud_pt3d.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera
    
    
	return true;
}
