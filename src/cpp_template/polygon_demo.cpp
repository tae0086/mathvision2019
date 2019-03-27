#include <iostream>
#include <cstdlib>

#include "polygon_demo.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

PolygonDemo::PolygonDemo()
{
	m_data_ready = false;
	data_size = 0;
}

PolygonDemo::~PolygonDemo()
{
}

bool isPositive(int val) {
	return val > 0;
}

bool isNegative(int val) {
	return val < 0;
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
				circle(frame, center, (int)(radius + 0.5), Scalar(0, 255, 0), 1);
				circle(frame, center, 2, Scalar(0, 255, 0), CV_FILLED);
			}
		}
	}

	imshow("PolygonDemo", frame);
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

	// absolute value
	if (area < 0.0) {
		area *= -1;
	}

	return area;
}

// return true if pt is interior point
bool PolygonDemo::ptInPolygon(const std::vector<cv::Point>& vtx, Point pt)
{
	return false;
}

// return homography type: NORMAL, CONCAVE, TWIST, REFLECTION, CONCAVE_REFLECTION
int PolygonDemo::classifyHomography(const std::vector<cv::Point>& pts1, const std::vector<cv::Point>& pts2)
{
	if (pts1.size() != 4 || pts2.size() != 4) return -1;
	vector<int> values;

	// cross producct in adjacent lines
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
		//putText(frame, to_string(jp * jq / 10000), Point(pts2[i].x + 20, pts2[i].y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
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

// estimate a circle that best approximates the input points and return center and radius of the estimate circle
bool PolygonDemo::fitCircle(const std::vector<cv::Point>& pts, cv::Point2d& center, double& radius)
{
	int n = (int)pts.size();
	if (n < 3) return false;
	return false;
}

void PolygonDemo::drawPolygon(Mat& frame, const std::vector<cv::Point>& vtx, bool closed)
{
	int i = 0;
	for (i = 0; i < (int)m_data_pts.size(); i++)
	{
		circle(frame, m_data_pts[i], 2, Scalar(255, 255, 255), CV_FILLED);
		string vtx(1, 'A' + i);
		string vtx_text = vtx + "(" + to_string(m_data_pts[i].x) + ", " + to_string(m_data_pts[i].y) + ")";
		putText(frame, vtx_text, Point(m_data_pts[i].x + 10, m_data_pts[i].y + 10), FONT_HERSHEY_SIMPLEX, .6, Scalar(255, 255, 255));
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
