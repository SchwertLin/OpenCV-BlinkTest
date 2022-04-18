#include <dlib\opencv.h>
#include <opencv2\opencv.hpp>
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\image_processing\render_face_detections.h>
#include <dlib\image_processing.h>
#include <dlib\gui_widgets.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>

using namespace std;
using namespace dlib;
using namespace cv;

//存储眼睛的上一个点的坐标
int eye_pre_x = 20;					//原点横坐标
int eye_pre_y = 400;				//原点纵坐标
int eye_now_x = 20;	
int eye_now_y = 400;
int START_TIME = 0, END_TIME = 0;
//存储眨眼的次数
unsigned int count_blink = 0;		//眨眼次数--每次眨眼EAR都要经历从  >0.2-<0.2->0.2 的过程
double blink_EAR_before =0.0;		//眨眼前
double blink_EAR_now =1.0;			//眨眼中
double blink_EAR_after = 0.0;		//眨眼后

Mat Eye_Waveform(420, 420, CV_8UC3, Scalar(255, 255, 255));//用于记录眨眼的波形图--白色

double GetEAR(std::vector<full_object_detection> shapes) {
	//获取左右眼的6点坐标（共12个点）
	//点36的坐标
	unsigned int x_36 = shapes[0].part(36).x();
	unsigned int y_36 = shapes[0].part(36).y();
	//点37的坐标
	unsigned int x_37 = shapes[0].part(37).x();
	unsigned int y_37 = shapes[0].part(37).y();
	//点38的坐标
	unsigned int x_38 = shapes[0].part(38).x();
	unsigned int y_38 = shapes[0].part(38).y();
	//点39的坐标
	unsigned int x_39 = shapes[0].part(39).x();
	unsigned int y_39 = shapes[0].part(39).y();
	//点40的坐标
	unsigned int x_40 = shapes[0].part(40).x();
	unsigned int y_40 = shapes[0].part(40).y();
	//点41的坐标
	unsigned int x_41 = shapes[0].part(41).x();
	unsigned int y_41 = shapes[0].part(41).y();
	//点42的坐标
	unsigned int x_42 = shapes[0].part(42).x();
	unsigned int y_42 = shapes[0].part(42).y();
	//点37的坐标---------------------------------------
	unsigned int x_43 = shapes[0].part(43).x();
	unsigned int y_43 = shapes[0].part(43).y();
	//点38的坐标
	unsigned int x_44 = shapes[0].part(44).x();
	unsigned int y_44 = shapes[0].part(44).y();
	//点39的坐标
	unsigned int x_45 = shapes[0].part(45).x();
	unsigned int y_45 = shapes[0].part(45).y();
	//点40的坐标
	unsigned int x_46 = shapes[0].part(46).x();
	unsigned int y_46 = shapes[0].part(46).y();
	//点41的坐标
	unsigned int x_47 = shapes[0].part(47).x();
	unsigned int y_47 = shapes[0].part(47).y();
	//计算EAR
	int left_h1 = y_41 - y_37;			 //37到41的纵向距离
	int left_h2 = y_40 - y_38;			 //38到40的纵向距离
	double left_h = (left_h1+left_h2) / 2.0;//眼睛上下距离
	int left_w = x_39 - x_36;
	if (left_h == 0)left_h = 1;//当眼睛闭合的时候，距离可能检测为0，宽高比出错
	double left_EAR = left_h / left_w;		//眼睛宽高比

	int right_h1 = y_47 - y_43;			 //37到41的纵向距离
	int right_h2 = y_46 - y_44;			 //38到40的纵向距离
	double right_h = (right_h1 + right_h2) / 2.0;//眼睛上下距离
	int right_w = x_45 - x_42;
	if (right_h == 0)right_h = 1;//当眼睛闭合的时候，距离可能检测为0，宽高比出错
	double right_EAR = right_h / right_w;		//眼睛宽高比

	//取两只眼睛的平均宽高比作为眼睛的宽高比
	double EAR = (left_EAR + right_EAR) / 2.0;
	return EAR;
}

void Draw_init() {//画初始坐标轴
	Point p1 = Point(20, 400);
	Point p2 = Point(20, 0);
	Point p3 = Point(420, 400);
	//画线
	cv::line(Eye_Waveform, Point(20, 200), Point(420, 200), Scalar(0, 16, 219), 1, LINE_AA);
	cv::line(Eye_Waveform, p1, p2, Scalar(219, 22, 0), 1, LINE_AA);//蓝色的线
	cv::line(Eye_Waveform, p1, p3, Scalar(219, 22, 0), 1, LINE_AA);
	//坐标--硬写实现
	cv::putText(Eye_Waveform, "0", Point(10, 410), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);//源点
	cv::putText(Eye_Waveform, "0.1", Point(0, 310), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "0.2", Point(0, 210), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "0.3", Point(0, 110), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "0.4", Point(0, 10), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	//坐标点
	cv::circle(Eye_Waveform, Point(20, 400), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 300), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 200), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 100), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 0), 1, Scalar(0, 0, 0), 1);
	//附加文字
	cv::putText(Eye_Waveform, "Picture1", Point(200, 410), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "time", Point(380, 410), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
}

void Count(double EAR) {//传入参数是目前计算得出的EAR
	if (blink_EAR_before < EAR)blink_EAR_before = EAR; 
	if (blink_EAR_now > EAR)blink_EAR_now = EAR; 
	if (blink_EAR_after < EAR)blink_EAR_after = EAR;
	if ((blink_EAR_before > 0.2) && (blink_EAR_now <= 0.2) && (blink_EAR_after > 0.2)) {//出现的问题是所有的函数都进入这里
		END_TIME = clock();
		cout << "EAR：" << EAR << " 前：" << blink_EAR_before << " 中：" << blink_EAR_now << " 后：" << blink_EAR_after << endl;
		blink_EAR_before = 0.0;
		blink_EAR_now = 1.0;
		blink_EAR_after = 0.0;
		if ((END_TIME - START_TIME) / CLOCKS_PER_SEC * 1000 > 200)count_blink++;
		START_TIME = clock();
	}
}//【对这里进行一个计时的操作。】

void Draw_now(double EAR) {
	Count(EAR);//顺便计算一下cnt
	eye_now_x = eye_now_x + 1;			//横坐标（每10个像素描一个点）
	eye_now_y = 400*(1-EAR/0.4);		//纵坐标
	Point pos1 = Point(eye_pre_x, eye_pre_y);//上一个点
	Point pos2 = Point(eye_now_x, eye_now_y);//现在的点
	cv::line(Eye_Waveform, pos1, pos2, Scalar(0, 0, 0), 1, LINE_AA);//画线-黑色
	eye_pre_x = eye_now_x;
	eye_pre_y = eye_now_y;
}

string DoubleToString(double num) {
	stringstream oss;
	oss << num;
	return oss.str();
}

int main() {
	Draw_init();
	//打开摄像头
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "【无法打开摄像头】" << endl;
		return 1;
	}

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pos_model;
	deserialize("E:/CV/shape_predictor_68_face_landmarks.dat")>>pos_model;

	while (true) {
		Mat frame;
		cap >> frame;
		cv_image<bgr_pixel> cimg(frame);//将图像转化为dlib中的BGR格式

		std::vector<dlib::rectangle> faces = detector(cimg);
		std::vector<full_object_detection> shapes;
		for (unsigned int i = 0; i < faces.size(); i++)shapes.push_back(pos_model(cimg, faces[i]));

		if (!shapes.empty()) {
			for (int j = 0; j < shapes.size(); j++) {
				for (int i = 0; i < 68; i++) {//用来画特征值的点
					cv::circle(frame, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 2, cv::Scalar(219,22,0), -1);
				}
			}
			Draw_now(GetEAR(shapes));//获得EAR并作为参数画出眨眼波形图
			//把hight_left_eye从float类型转化成字符串类型
			string count_blink_text=DoubleToString(count_blink);
			count_blink_text = "Count:" + count_blink_text;
			putText(frame, count_blink_text, Point(10, 100), FONT_HERSHEY_COMPLEX, 1.0, Scalar(0, 0, 0), 1, LINE_AA);
		}
		cv::imshow("Dlib mark", frame);
		cv::imshow("Blink waveform figure", Eye_Waveform);
		if (waitKey(30) ==27) break;
	}
	destroyAllWindows();
}