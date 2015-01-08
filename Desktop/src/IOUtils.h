#include <iostream>
#include <fstream>
#include <cassert>
#include <exception>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using namespace std;
using namespace cv;

struct ImageDesc {
    std::string path;
    Mat rotationMatrix;
    double pitch, azimuth, roll;
    double globalPosX, globalPosY, globalPosZ;


    ImageDesc() {
        pitch = azimuth = roll = 0;
        globalPosX = globalPosY = globalPosZ = 0;
    }
};

void parseCameraInstersincParameters(char const *filename, Mat &K, Mat &distCoeffs){
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    fs["camera_matrix"] >> K;
    fs["distortion_coefficients"] >> distCoeffs;
    K.convertTo(K, CV_64FC1);
    distCoeffs.convertTo(distCoeffs, CV_64FC1);
}


void parseSensorFile(char const *filename, vector<ImageDesc> &imageDescriptions) {
    try {
        boost::property_tree::ptree pt;
        boost::property_tree::read_json(filename, pt);

        cout << pt.size() << endl;


        BOOST_FOREACH(boost::property_tree::ptree::value_type & v, pt) {
            ImageDesc newImageDesc;
            assert(v.first.empty()); // array elements have no names
//                        std::cout << v.first.data() << std::endl;
            std::string path = v.second.get<std::string>("photoPath", "none");
//                        std::cout << path << std::endl;
            newImageDesc.path = path;

            Mat matrix(16, 1, CV_64FC1);
            int i = 0;
            BOOST_FOREACH(boost::property_tree::ptree::value_type & child,
                    v.second.get_child("rotationMatrix")) {
                float value = child.second.get<double>("");
                matrix.at<double>(i) = value;
                i++;
            }

            matrix = matrix.reshape(1, 4);
//                        std::cout << matrix << std::endl;
            newImageDesc.rotationMatrix = (matrix);
            newImageDesc.azimuth = v.second.get<double>("azimuth", 0);
            newImageDesc.roll = v.second.get<double>("roll", 0);
            newImageDesc.pitch = v.second.get<double>("pitch", 0);
            newImageDesc.globalPosX = v.second.get<double>("posX", 0);
            newImageDesc.globalPosY = v.second.get<double>("posY", 0);
            newImageDesc.globalPosZ = v.second.get<double>("posZ", 0);
            imageDescriptions.push_back(newImageDesc);
            // etc
        }
    }
    catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
    }
}

void savePointList(string fname, Mat &points) {

    // open file for write
    fstream file(fname.c_str(), ios::out);
    if (!file) {
        cerr << "ERROR: cannot open file " << fname << endl;
        return;
    }

    // if homogeneous points: norm and write points
    if (points.rows == 4) {
        for (int i = 0; i < points.cols; i++) {
            file << points.at<double>(0, i) / points.at<double>(3, i) << "," << points.at<double>(1, i) / points.at<double>(3, i) << "," << points.at<double>(2, i) / points.at<double>(3, i) << endl;
        }
    }
    // if euclidian points: write points
    if (points.rows == 3) {
        for (int i = 0; i < points.cols; i++) {
            file << points.at<double>(0, i) << "," << points.at<double>(1, i) << "," << points.at<double>(2, i) << endl;
        }
    }

    // close file
    file.close();
}

void savePointList(string fname, vector<Point3d> &points) {

    // open file for write
    fstream file(fname.c_str(), ios::out);
    if (!file) {
        cerr << "ERROR: cannot open file " << fname << endl;
        return;
    }


    // if euclidian points: write points
    for (int i = 0; i < points.size(); i++) {
        file << points[i].x << "," << points[i].y << "," << points[i].z << endl;
    }

    // close file
    file.close();
}

void getRelativeMatrixAndAngles(vector<ImageDesc> &imageDescriptions, int prevIdx, int nextIdx, Mat &rotDiffGlobal, double &pitchDiff, double &azimuthDiff, double &rollDiff);

void getRelativeMatrixAndAngles(vector<ImageDesc> &imageDescriptions, int prevIdx, int nextIdx, Mat &rotDiffGlobal, double &pitchDiff, double &azimuthDiff, double &rollDiff) {
    double pitch1, azimuth1, roll1;
    double pitch2, azimuth2, roll2;
    pitch1 = (imageDescriptions[prevIdx].pitch);
    azimuth1 = (imageDescriptions[prevIdx].azimuth);
    roll1 = (imageDescriptions[prevIdx].roll);
    pitch2 = (imageDescriptions[nextIdx].pitch);
    azimuth2 = (imageDescriptions[nextIdx].azimuth);
    roll2 = (imageDescriptions[nextIdx].roll);

    pitchDiff = radians((pitch2 - pitch1));
    azimuthDiff = radians((azimuth2 - azimuth1));
//TODO according to Android app bug we need to minus this one
//    rollDiff = radians(-(roll2 - roll1));
//    rotDiffGlobal = getRotated3DMatrix<double>(pitchDiff, azimuthDiff, rollDiff);
    rollDiff = radians(-(roll2 - roll1));
    rotDiffGlobal = getRotation3DMatrix<double>(pitchDiff, azimuthDiff, rollDiff);
}
