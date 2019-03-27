#include "windows.h"
#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#ifdef _DEBUG
#pragma comment(lib,"opencv_world320d.lib")
#else
#pragma comment(lib,"opencv_world320.lib")
#endif

using namespace std;
using namespace cv;


enum { NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION };

bool isPositive(int val) {
	return val > 0;
}

bool isNegative(int val) {
	return val < 0;
}

int checkHomography(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2)
{
	if (pts1.size() != 4 || pts2.size() != 4) return -1;
	vector<int> values;

	// cross product in adjacent lines
	for (int i = 0; i < 4; i++) {
		// set h (before), j (next)
		int h = i - 1; if (h < 0) h = 3;
		int j = i + 1; if (j > 3) j = 0;

		// cross product in ORIGINAL
		Point p1 = pts1[h] - pts1[i];
		Point p2 = pts1[j] - pts1[i];
		int jp = p1.x * p2.y - p1.y * p2.x;

		// cross product in TRANSFORMED
		Point q1 = pts2[h] - pts2[i];
		Point q2 = pts2[j] - pts2[i];
		int jq = q1.x * q2.y - q1.y * q2.x;

		values.push_back(jp * jq);
		cout << (char)(65 + i) << " " << jp * jq / 10000 << endl;
	}

	// 1. reflection (ALL values < 0)
	if (count_if(values.begin(), values.end(), isNegative) == 4) {
		return REFLECTION;
	}
	// 2. twist (first or last TWO values < 0)
	else if (count_if(values.begin(), values.end(), isNegative) == 2) {
		return TWIST;
	}
	// 3. concave (ONLY concave value < 0)
	else if (count_if(values.begin(), values.end(), isNegative) == 1) {
		return CONCAVE;
	}
	// 4. concave reflection (ONLY concave value > 0)
	else if (count_if(values.begin(), values.end(), isPositive) == 1) {
		return CONCAVE_REFLECTION;
	}

	// 5. normal (ALL values > 0)
	return NORMAL;
}


void proc_video(VideoCapture& vc, Mat& model)
{
	int min_match_num = 4;
	Mat model_gray;
	cvtColor(model, model_gray, CV_BGR2GRAY);

	// local feature method
	Ptr<Feature2D> fd = ORB::create();

	// detect model key points
	vector<KeyPoint> obj_keypts;
	fd->detect(model_gray, obj_keypts);
	if ((int)obj_keypts.size() < min_match_num) return;

	// compute descriptor of the model key points
	Mat obj_descriptors;
	fd->compute(model_gray, obj_keypts, obj_descriptors);

	// descriptor matcher
	Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce-Hamming");
	descriptorMatcher->add(obj_descriptors);

	// main loop
	Mat frame;
	Mat gray;
	while (1)
	{
		// grab frame
		vc >> frame;
		if (frame.empty()) break;
		cvtColor(frame, gray, CV_BGR2GRAY);

		// detect keypoints
		vector<KeyPoint> img_keypts;
		fd->detect(gray, img_keypts);
		if ((int)img_keypts.size() < min_match_num) continue;

		// compute descriptor
		Mat img_descriptors;
		fd->compute(gray, img_keypts, img_descriptors);

		// match descriptors
		vector<DMatch> matches;
		descriptorMatcher->match(img_descriptors, matches);
		if ((int)matches.size() < min_match_num) continue;

		// find homography
		vector<Point2f> obj_pts, img_pts;
		for (int i = 0; i < (int)matches.size(); i++)
		{
			int idx1 = matches[i].trainIdx;
			int idx2 = matches[i].queryIdx;
			obj_pts.push_back(obj_keypts[idx1].pt);
			img_pts.push_back(img_keypts[idx2].pt);
		}
		cv::Mat inlier_mask = cv::Mat::zeros(matches.size(), 1, CV_8U);
		Mat H = findHomography(obj_pts, img_pts, RANSAC, 3.0, inlier_mask);

		// draw matches
		Mat result;
		drawMatches(frame, img_keypts, model, obj_keypts, matches, result, cv::Scalar(0, 0, 255), cv::Scalar(0, 127, 0), inlier_mask);

		// draw estimated homography transformation
		if (!H.empty())
		{
			// model rectangle
			int model_w = model.cols;
			int model_h = model.rows;

			vector<Point2f> corner_pts1;
			corner_pts1.push_back(Point(0, 0));
			corner_pts1.push_back(Point(0, model_h - 1));
			corner_pts1.push_back(Point(model_w - 1, model_h - 1));
			corner_pts1.push_back(Point(model_w - 1, 0));

			// transformed rectangle
			vector<Point2f> corner_pts2;
			perspectiveTransform(corner_pts1, corner_pts2, H);

			// check if the estimated homography is valid or not
			int homo_type = checkHomography(corner_pts1, corner_pts2);
			char type_str[100];
			switch (homo_type)
			{
			case NORMAL:
				sprintf_s(type_str, 100, "normal");
				break;
			case CONCAVE:
				sprintf_s(type_str, 100, "concave");
				break;
			case TWIST:
				sprintf_s(type_str, 100, "twist");
				break;
			case REFLECTION:
				sprintf_s(type_str, 100, "reflection");
				break;
			case CONCAVE_REFLECTION:
				sprintf_s(type_str, 100, "concave reflection");
				break;
			default:
				sprintf_s(type_str, 100, "none");
				break;
			}
			if (homo_type == NORMAL)
				putText(result, type_str, Point(15, 35), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);
			else
				putText(result, type_str, Point(15, 35), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

			// draw transformed rectangle
			for (int i = 0; i < 4; i++)
			{
				if (homo_type == NORMAL)
					line(result, corner_pts2[i], corner_pts2[(i + 1) % 4], Scalar(255, 0, 0), 2);
				else
					line(result, corner_pts2[i], corner_pts2[(i + 1) % 4], Scalar(0, 0, 255), 2);
			}
		}

		// display result
		imshow("match", result);
		char ch = waitKey(1);
		if (ch == 27) break;				// ESC Key
		else if (ch == 32)					// SPACE Key
		{
			while ((ch = waitKey(10)) != 32 && ch != 27);
			if (ch == 27) break;
		}
	}
}


int main(int argc, char* argv[])
{
	int model_i = 0;

	string model_path[] = { "data\\mousepad.bmp", "data\\blais.jpg" };
	string avi_path[] = { "data\\mousepad.mp4", "data\blais.mp4" };

	// load video data
	VideoCapture vc(avi_path[model_i]);
	if (!vc.isOpened())
	{
		cout << "can't open video file" << endl;
		return -1;
	}

	// load object model
	Mat model = imread(model_path[model_i]);
	if (model.empty())
	{
		cout << "fail to load model image!" << endl;
		return -1;
	}

	// process video
	proc_video(vc, model);
	destroyAllWindows();

	return 0;
}
