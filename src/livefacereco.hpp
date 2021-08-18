//
// Created by Xinghao Chen 2020/7/27
// Edited by M Rijal Al Fariz 2021/5
//
#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include "stdlib.h"
#include <iostream>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "arcface.h"

using namespace std;
using namespace cv;



// Adjustable Parameters
const bool largest_face_only=false;
const bool record_face=false;
const int distance_threshold = 10;
const int jump=10;

const float face_thre_def =0.40;
const float true_thre_def =0.60;
const int input_width_def = 640;
const int input_height_def = 360;
const int min_face_size_def = input_height_def*15/100;

const int output_width = 640;
const int output_height = 360;
const string project_path="/<YOUR_PATH>/face-cam-cpp";
//end

const cv::Size frame_size = Size(output_width,output_height);
const float ratio_x = (float)output_width/ input_width_def;
const float ratio_y = (float)output_height/ input_height_def;


int MTCNNDetection();


