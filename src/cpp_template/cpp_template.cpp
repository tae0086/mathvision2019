#include <ctime>

#include "windows.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "polygon_demo.hpp"

#ifdef _DEBUG
#pragma comment(lib,"opencv_world320d.lib")
#else
#pragma comment(lib,"opencv_world320.lib")
#endif

using namespace std;
using namespace cv;

void onMouse(int evt, int x, int y, int flags, void* param)
{
	PolygonDemo *p = (PolygonDemo *)param;
	p->handleMouseEvent(evt, x, y, flags);
}

void onButton(int state, void *data) {

}

int main(int argc, char* argv[])
{
	// seed
	srand(static_cast<unsigned int>(time(0)));

	// main window
	Mat frame = Mat::zeros(480, 640, CV_8UC3);
	putText(frame, "Input data points (double click: finish)", Point(10, 470), FONT_HERSHEY_SIMPLEX, .5, Scalar(0, 148, 0), 1);
	imshow("PolygonDemo", frame);

	// event handler 등록
	PolygonDemo tmp;
	setMouseCallback("PolygonDemo", onMouse, &tmp);

	// exit with any key input
	char ch = waitKey();
	return 0;
}
