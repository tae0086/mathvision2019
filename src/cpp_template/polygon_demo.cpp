#include <iostream>
#include <cstdlib>
#include <atomic>

#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

bool isPositive(int val)
{
	return val > 0;
}

bool isNegative(int val)
{
	return val < 0;
}

double toDegree(double radian)
{
	return radian * 180 / PI;
}

double cauchyWeight()
{
	return 0.0;
}

int getRandomNumber(int min, int max)
{
	const double fraction = 1.0 / (RAND_MAX + 1.0);
	return min + static_cast<int>((max - min + 1) * (rand() * fraction));
}

vector<int> getRandomNumbers(int min, int max, int num)
{
	vector<int> rnums;
	for (int i = 0; i < num; i++) {
		int rnum;
		do {
			rnum = getRandomNumber(min, max);
		} while (find(rnums.begin(), rnums.end(), rnum) != rnums.end()); // if new random number is already exists in vector, then regenerate the random number.
		rnums.push_back(rnum);
	}
	return rnums;
}

PolygonDemo::PolygonDemo()
{
	m_data_ready = false;
	data_size = 0;
}

PolygonDemo::~PolygonDemo() {}

void PolygonDemo::refreshWindow()
{
	int width = 640, height = 480;
	Mat frame = Mat::zeros(height, width, CV_8UC3);

	if (!m_data_ready)
		putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);

	drawPolygon(frame, m_data_pts, m_data_ready);
	if (m_data_ready)
	{
		// polygon area
		if (m_param.compute_area)
		{
			int area = polyArea(m_data_pts);
			char str[100];
			sprintf_s(str, 100, "area = %d", area);
			putText(frame, str, Point(10, height - 20), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
		}

		// pt in polygon
		if (m_param.check_ptInPoly)
		{
			for (int i = 0; i < (int)m_test_pts.size(); i++)
			{
				if (ptInPolygon(m_data_pts, m_test_pts[i]))
				{
					circle(frame, m_test_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				}
				else
				{
					circle(frame, m_test_pts[i], 2, Scalar(128, 128, 128), CV_FILLED);
				}
			}
		}

		// homography check
		if (m_param.check_homography && data_size == 4)
		{
			// rect points
			int rect_sz = 100;
			vector<Point> rc_pts;
			rc_pts.push_back(Point(0, 0));
			rc_pts.push_back(Point(0, rect_sz));
			rc_pts.push_back(Point(rect_sz, rect_sz));
			rc_pts.push_back(Point(rect_sz, 0));
			rectangle(frame, Rect(0, 0, rect_sz, rect_sz), Scalar(255, 255, 255), 1);

			// draw mapping
			//char* abcd[4] = { "A", "B", "C", "D" };
			for (int i = 0; i < 4; i++)
			{
				line(frame, rc_pts[i], m_data_pts[i], Scalar(255, 0, 0), 1);
				circle(frame, rc_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				circle(frame, m_data_pts[i], 2, Scalar(0, 255, 0), CV_FILLED);
				//putText(frame, abcd[i], m_data_pts[i], FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
			}

			// check homography
			int homo_type = classifyHomography(rc_pts, m_data_pts);
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
			}

			putText(frame, type_str, Point(10, height - 50), FONT_HERSHEY_SIMPLEX, .8, Scalar(0, 255, 255), 1);
		}

		// fit circle
		if (m_param.fit_circle)
		{
			Point2d center;
			double radius = 0;
			bool ok = fitCircle(m_data_pts, center, radius);
			if (ok)
			{
				string crd_text = "(" + to_string((int)center.x) + ", " + to_string((int)center.y) + ")";
				string radius_text = "r = " + to_string((int)radius);
				circle(frame, center, (int)(radius + 0.5), Scalar(227, 195, 74), 2);
				circle(frame, center, 2, Scalar(227, 195, 74), CV_FILLED);
				putText(frame, crd_text, Point((int)center.x - 50, (int)center.y - 10), FONT_HERSHEY_SIMPLEX, .6, Scalar(227, 195, 74));
				putText(frame, radius_text, Point(10, height - 20), FONT_HERSHEY_SIMPLEX, .6, Scalar(227, 195, 74));
			}
		}

		// fit ellipse
		if (m_param.fit_ellipse) {
			Point2d m;
			Point2d v;
			double theta;
			bool ok = fitEllipse(m_data_pts, m, v, theta);
			if (ok) {
				string crd_text = "(" + to_string((int)m.x) + ", " + to_string((int)m.y) + ")";
				string radius_text = "r = (" + to_string((int)v.x) + ", " + to_string((int)v.y) + ")";
				try{
					ellipse(frame, m, Size((int)v.x, (int)v.y), theta, 0, 360, Scalar(162, 156, 248), 2);
				}
				catch (Exception e) {
				}
				circle(frame, m, 2, Scalar(162, 156, 248), CV_FILLED);
				putText(frame, crd_text, Point((int)m.x - 50, (int)m.y - 10), FONT_HERSHEY_SIMPLEX, .6, Scalar(162, 156, 248));
				putText(frame, radius_text, Point(10, height - 20), FONT_HERSHEY_SIMPLEX, .6, Scalar(162, 156, 248));
			}
		}

		// fit line
		if (m_param.fit_line) {
			// y = ax + b
			/*bool ok1 = fitLine(m_data_pts, pt1, pt2, false);
			if (ok1) {
			line(frame, pt1, pt2, Scalar(0, 255, 0));
			putText(frame, "y = ax + b", Point(10, 20), FONT_HERSHEY_SIMPLEX, .6, Scalar(0, 255, 0));
			}*/

			// ax + by + c = 0
			/*bool ok3 = fitLine(m_data_pts, pt1, pt2, true);
			if (ok3) {
			line(frame, pt1, pt2, Scalar(0, 0, 255));
			putText(frame, "ax + by + c = 0", Point(10, 40), FONT_HERSHEY_SIMPLEX, .6, Scalar(0, 0, 255));
			}*/

			// robust version with cauchy weight function
			//Mat p = Mat::zeros(2, 1, CV_64F); // for paramter (a, b)
			//vector<Scalar> colors{ Scalar(0, 222, 229), Scalar(1, 200, 231), Scalar(2, 178, 234), Scalar(3, 156, 237), Scalar(4, 134, 240), // BGR color for gradation (from YELLOW to RED)
			//	Scalar(5, 112, 243), Scalar(6, 90, 246), Scalar(7, 68, 249), Scalar(8, 46, 252), Scalar(10, 25, 255) };

			//for (int i = 0; i < 10; i++) {
			//	bool ok2 = fitRobustLine(m_data_pts, pt1, pt2, p, i);
			//	if (ok2) {
			//		line(frame, pt1, pt2, colors[i]);
			//	}
			//}

			// RANSAC
			Point2d pt1, pt2;
			vector<Point2d> inliers;
			if (fitRANSACLine(m_data_pts, pt1, pt2, inliers)) {
				line(frame, pt1, pt2, Scalar(0, 255, 0));
				for each (Point2d inlier in inliers) {
					circle(frame, inlier, 1, Scalar(0, 255, 0), 2);
				}
			}
		}
	}

	imshow("PolygonDemo", frame);
}

bool PolygonDemo::fitRANSACLine(const std::vector<cv::Point>& pts, cv::Point2d& pt1, cv::Point2d& pt2, vector<cv::Point2d>& inliers_max)
{
	int n = pts.size();
	if (n < 2) {
		cout << "The number of points is less than 2." << endl;
		return false;
	}

	// threshold and number of iteration
	const int T = 10;
	const int MAX_ITER = 100;

	Mat x_max = Mat::zeros(3, 1, CV_64F);

	for (int iter = 0; iter < MAX_ITER; iter++) {
		// random two extractions for line
		vector<int> indices = getRandomNumbers(0, n - 1, 2);
		vector<cv::Point> points_in_line;

		for (int i = 0; i < (int)indices.size(); i++) {
			//cout << "index " << indices[i] << endl;
			points_in_line.push_back(pts[indices[i]]);
		}

		// line: ax + by + c = 0
		Mat A = Mat::ones(2, 3, CV_64F);
		A.at<double>(0, 0) = points_in_line[0].x;
		A.at<double>(0, 1) = points_in_line[0].y;
		A.at<double>(1, 0) = points_in_line[1].x;
		A.at<double>(1, 1) = points_in_line[1].y;

		// singular value decomposition of A
		Mat w, u, vt;
		SVD::compute(A, w, u, vt, SVD::FULL_UV);

		// solution of equation
		Mat x = vt.row(vt.rows - 1);
		double a = x.at<double>(0, 0);
		double b = x.at<double>(0, 1);
		double c = x.at<double>(0, 2);
		//cout << "a: " << a << ", b: " << b << ", c: " << c << endl;

		// count inliers
		vector<Point2d> inliers;

		for (int i = 0; i < n; i++) {
			int x = pts[i].x;
			int y = pts[i].y;
			int y_hat = (int)(-(a * x + c) / b);
			int residual = y_hat - y;

			if (abs(residual) < T) {
				inliers.push_back(pts[i]);
			}
		}

		if (inliers_max.size() < inliers.size()) {
			inliers_max = inliers;
			//x_max = x;
			x_max.at<double>(0, 0) = x.at<double>(0, 0);
			x_max.at<double>(1, 0) = x.at<double>(0, 1);
			x_max.at<double>(2, 0) = x.at<double>(0, 2);
		}
	}

	double a_max = x_max.at<double>(0, 0);
	double b_max = x_max.at<double>(1, 0);
	double c_max = x_max.at<double>(2, 0);
	cout << "a_max: " << a_max << ", b_max: " << b_max << ", c_max: " << c_max << endl;

	// point 1
	pt1.x = 0;
	pt1.y = -((a_max * pt1.x + c_max) / b_max);

	// point 2
	pt2.x = 640;
	pt2.y = -((a_max * pt2.x + c_max) / b_max);

	return true;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
	int i = 0;
	for (i = 0; i < (int)m_data_pts.size(); i++)
	{
		circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
		//string vtx(1, 'A' + i);
		//string vtx_text = vtx + "(" + to_string(m_data_pts[i].x) + ", " + to_string(m_data_pts[i].y) + ")";
		//putText(frame, vtx_text, Point(m_data_pts[i].x + 10, m_data_pts[i].y + 10), FONT_HERSHEY_SIMPLEX, .6, Scalar(255, 255, 255));
	}
	for (i = 0; i < (int)m_data_pts.size() - 1; i++)
	{
		//line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
	}
	if (closed)
	{
		//line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
	}
}

// return the area of polygon
int PolygonDemo::polyArea(const std::vector<cv::Point>& vtx)
{
	int area = 0;

	// cross product
	int n = (int)vtx.size();
	for (int i = 0; i < n; i++) {
		area += (vtx[i].x * vtx[(i + 1) % n].y - vtx[i].y * vtx[(i + 1) % n].x) / 2;
	}

	// absolue value because area cannot be negative
	return abs(area);
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	if (pts1.size() != 4 || pts2.size() != 4) return -1;
	vector<int> values;

	// cross producct in adjacent lines
	for (int i = 0; i < 4; i++) {
		// set h (before), j (next)
		int h = (i - 1 + 4) % 4;
		int j = (i + 1) % 4;

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
		//putText(frame, to_string(jp * jq / 10000), Point(pts2[i].x + 20, pts2[i].y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
	}

	// 1. reflection (ALL values < 0)
	if (count_if(values.begin(), values.end(), isNegative) == 4)
		return REFLECTION;
	// 2. twist (first or last TWO values < 0)
	else if (count_if(values.begin(), values.end(), isNegative) == 2)
		return TWIST;
	// 3. concave (ONLY concave value < 0)
	else if (count_if(values.begin(), values.end(), isNegative) == 1)
		return CONCAVE;
	// 4. concave reflection (ONLY concave value > 0)
	else if (count_if(values.begin(), values.end(), isPositive) == 1)
		return CONCAVE_REFLECTION;

	// 5. normal (ALL values > 0)
	return NORMAL;
}

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	int n = (int)pts.size();
	if (n < 3) return false;

	// initialize A and b using pts
	Mat A = Mat::ones(n, 3, CV_64F);
	Mat b_vec = Mat(n, 1, CV_64F);

	for (int i = 0; i < n; i++) {
		int x = pts[i].x;
		int y = pts[i].y;

		A.at<double>(i, 0) = x;
		A.at<double>(i, 1) = y;
		b_vec.at<double>(i, 0) = -pow(x, 2) - pow(y, 2);
	}
	//cout << "A: " << A << endl;
	//cout << "B: " << B << endl;

	// inverse or pseudo inverse of A
	Mat A_inv;
	invert(A, A_inv, DECOMP_SVD);
	//cout << "inverse of A: " << A_inv << endl;

	// solution of equation
	Mat x = A_inv * b_vec;
	double a = x.at<double>(0, 0);
	double b = x.at<double>(0, 1);
	double c = x.at<double>(0, 2);
	//cout << "x: " << x << endl;
	//cout << "a: " << a << endl;
	//cout << "b: " << b << endl;
	//cout << "c: " << c << endl;

	// center and radius of circle
	center.x = -a / 2;
	center.y = -b / 2;
	radius = sqrt(pow(a, 2) + pow(b, 2) - 4 * c) / 2.;

	// print information for circle
	cout << "* center: (" << center.x << ", " << center.y << ")" << endl;
	cout << "* radius: " << radius << endl;

	return true;
}

// estimate a ellipse that best approximates the input points and return center, radius, angle of the estimate ellipse
bool PolygonDemo::fitEllipse(const std::vector<cv::Point>& pts, cv::Point2d& m, cv::Point2d& v, double& theta) {
	int n = (int)pts.size();
	if (n < 5) return false;

	// initialize A and b using pts
	Mat A = Mat::ones(n, 6, CV_64F);

	for (int i = 0; i < n; i++) {
		int x = pts[i].x;
		int y = pts[i].y;

		A.at<double>(i, 0) = pow(x, 2);
		A.at<double>(i, 1) = x * y;
		A.at<double>(i, 2) = pow(y, 2);
		A.at<double>(i, 3) = x;
		A.at<double>(i, 4) = y;
	}

	// singular value decomposition of A
	Mat w, u, vt;
	SVD::compute(A, w, u, vt, SVD::FULL_UV);

	// solution of equation
	Mat x = vt.row(vt.rows - 1);
	double a = x.at<double>(0, 0); 	double b = x.at<double>(0, 1);
	double c = x.at<double>(0, 2); 	double d = x.at<double>(0, 3);
	double e = x.at<double>(0, 4); 	double f = x.at<double>(0, 5);

	// the angle between x-axis and major axis
	theta = (1. / 2.) * atan2(b, a - c);

	// the center of the ellipse
	m.x = (2 * c * d - b * e) / (pow(b, 2) - 4 * a * c);
	m.y = (2 * a * e - b * d) / (pow(b, 2) - 4 * a * c);

	// the semi-major axis length
	v.x = sqrt((a * pow(m.x, 2) + b * m.x * m.y + c * pow(m.y, 2) - f) / (a * pow(cos(theta), 2) + b * cos(theta) * sin(theta) + c * pow(sin(theta), 2)));
	// the semi-minor axis length
	v.y = sqrt((a * pow(m.x, 2) + b * m.x * m.y + c * pow(m.y, 2) - f) / (a * pow(sin(theta), 2) - b * cos(theta) * sin(theta) + c * pow(cos(theta), 2)));

	// convert radian to degree
	theta = toDegree(theta);

	// print information for ellipse
	cout << "* center: (" << m.x << ", " << m.y << ")" << endl;
	cout << "* length of MAJOR axis: " << v.x << endl;
	cout << "* length of MINOR axis: " << v.y << endl;
	cout << "* angle(degree): " << theta << endl;

	return true;
}

// if use_y_coef is true, SVD is used; otherwise, pseudo-inverse is used.
bool PolygonDemo::fitLine(const std::vector<cv::Point>& pts, cv::Point2d& pt1, cv::Point2d& pt2, bool use_y_coef)
{
	// number of input points
	int n = pts.size();

	// initialize A and b using pts
	Mat A;
	Mat b_vec;

	if (use_y_coef) { // ax + by + c = 0
		A = Mat::ones(n, 3, CV_64F);
		b_vec = Mat::zeros(n, 1, CV_64F);

		for (int i = 0; i < n; i++) {
			int x = pts[i].x;
			int y = pts[i].y;

			A.at<double>(i, 0) = x;
			A.at<double>(i, 1) = y;
		}
	}
	else { // y = ax + b
		A = Mat::ones(n, 2, CV_64F);
		b_vec = Mat(n, 1, CV_64F);

		for (int i = 0; i < n; i++) {
			int x = pts[i].x;
			int y = pts[i].y;

			A.at<double>(i, 0) = x;
			b_vec.at<double>(i, 0) = y;
		}
	}

	//cout << "A: " << A << endl;
	//cout << "b_vec: " << b_vec << endl;

	if (use_y_coef) { // ax + by + c = 0
		// singular value decomposition of A
		Mat w, u, vt;
		SVD::compute(A, w, u, vt, SVD::FULL_UV);

		// solution of equation
		Mat x = vt.row(vt.rows - 1);
		double a = x.at<double>(0, 0);
		double b = x.at<double>(0, 1);
		double c = x.at<double>(0, 2);

		// first point in line
		pt1.x = 0;
		pt1.y = -((a * pt1.x + c) / b);
		// second point in line
		pt2.x = 640;
		pt2.y = -((a * pt2.x + c) / b);
	}
	else { // y = ax + b
		// pseudo inverse of A
		Mat A_inverse;
		invert(A, A_inverse, DECOMP_SVD);
		//cout << "A_inverse: " << A_inverse << endl;

		// solution of equation
		Mat x = A_inverse * b_vec;
		//cout << "x: " << x << endl;

		double a = x.at<double>(0, 0);
		double b = x.at<double>(1, 0);
		//cout << "a: " << a << endl;
		//cout << "b: " << b << endl;

		// first point in line
		pt1.x = 0;
		pt1.y = a * pt1.x + b;
		// second point in line
		pt2.x = 640;
		pt2.y = a * pt2.x + b;
	}

	cout << "pt1: (" << pt1.x << ", " << pt1.y << ")" << endl;
	cout << "pt2: (" << pt2.x << ", " << pt2.y << ")" << endl;

	return true;
}

// robust parameter estimation using weighted least squares.
bool PolygonDemo::fitRobustLine(const std::vector<cv::Point>& pts, cv::Point2d& pt1, cv::Point2d& pt2, cv::Mat& p, int iteration)
{
	// n < 2 is not allowed.
	int n = pts.size();
	if (n < 2) return false;

	// line: y = ax + b
	// initialize A, b (Ax = b)
	Mat A = Mat::ones(n, 2, CV_64F);
	Mat y_vec = Mat::zeros(n, 1, CV_64F);

	for (int i = 0; i < n; i++) {
		int x = pts[i].x;
		int y = pts[i].y;
		A.at<double>(i, 0) = x;
		y_vec.at<double>(i, 0) = y;
	}

	if (iteration == 0) { // parameter initialization with least squares
		Mat A_pinv;
		invert(A, A_pinv, DECOMP_SVD);
		p = A_pinv * y_vec;
		//cout << "p: " << p << endl;
	}
	else {
		Mat r = A * p - y_vec;
		Mat w_vec = 1. / (abs(r) / 1.3998 + 1.); // main diagonal of matrix W
		Mat W = Mat::zeros(n, n, CV_64F);
		for (int i = 0; i < n; i++) {
			W.at<double>(i, i) = w_vec.at<double>(i, 0);
		}

		Mat M = A.t() * W * A;
		Mat M_inv;
		invert(M, M_inv, DECOMP_SVD);
		p = M_inv * A.t() * W * y_vec;
	}

	// two points for drawing line
	double a = p.at<double>(0, 0);
	double b = p.at<double>(1, 0);
	// first point
	pt1.x = 0;
	pt1.y = a * pt1.x + b;
	// second point
	pt2.x = 640;
	pt2.y = a * pt2.x + b;

	return true;
}

bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, cv::Point pt) {
	return false;
}

void PolygonDemo::handleMouseEvent(int evt, int x, int y, int flags)
{
	if (evt == CV_EVENT_LBUTTONDOWN)
	{
		if (!m_data_ready)
		{
			//cout<< ++data_size << "[" << x << ", " << y << "]" << endl;
			data_size++;
			m_data_pts.push_back(Point(x, y));
		}
		else
		{
			m_test_pts.push_back(Point(x, y));
		}
		refreshWindow();
	}
	else if (evt == CV_EVENT_LBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_LBUTTONDBLCLK)
	{
		cout << endl;
		m_data_ready = true;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONDBLCLK)
	{
	}
	else if (evt == CV_EVENT_MOUSEMOVE)
	{
	}
	else if (evt == CV_EVENT_RBUTTONDOWN)
	{
		m_data_pts.clear();
		m_test_pts.clear();
		m_data_ready = false;
		data_size = 0;
		refreshWindow();
	}
	else if (evt == CV_EVENT_RBUTTONUP)
	{
	}
	else if (evt == CV_EVENT_MBUTTONDOWN)
	{
	}
	else if (evt == CV_EVENT_MBUTTONUP)
	{
	}

	if (flags&CV_EVENT_FLAG_LBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_RBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_MBUTTON)
	{
	}
	if (flags&CV_EVENT_FLAG_CTRLKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_SHIFTKEY)
	{
	}
	if (flags&CV_EVENT_FLAG_ALTKEY)
	{
	}
}
