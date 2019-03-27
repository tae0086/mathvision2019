#include <iostream>

#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

bool isPositive(int val) {
	return val > 0;
}

bool isNegative(int val) {
	return val < 0;
}

double toDegree(double radian) {
	return radian * 180 / PI;
}

PolygonDemo::PolygonDemo()
{
	m_data_ready = false;
	data_size = 0;
}

PolygonDemo::~PolygonDemo()
{
}

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
				ellipse(frame, m, Size((int)v.x, (int)v.y), theta, 0, 360, Scalar(162, 156, 248), 2);
				circle(frame, m, 2, Scalar(162, 156, 248), CV_FILLED);
				putText(frame, crd_text, Point((int)m.x - 50, (int)m.y - 10), FONT_HERSHEY_SIMPLEX, .6, Scalar(162, 156, 248));
				putText(frame, radius_text, Point(10, height - 20), FONT_HERSHEY_SIMPLEX, .6, Scalar(162, 156, 248));
			}
		}
	}

	imshow("PolygonDemo", frame);
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
		line(frame, m_data_pts[i], m_data_pts[i + 1], Scalar(255, 255, 255), 1);
	}
	if (closed)
	{
		line(frame, m_data_pts[i], m_data_pts[0], Scalar(255, 255, 255), 1);
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
