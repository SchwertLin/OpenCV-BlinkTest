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

//�洢�۾�����һ���������
int eye_pre_x = 20;					//ԭ�������
int eye_pre_y = 400;				//ԭ��������
int eye_now_x = 20;	
int eye_now_y = 400;
int START_TIME = 0, END_TIME = 0;
//�洢գ�۵Ĵ���
unsigned int count_blink = 0;		//գ�۴���--ÿ��գ��EAR��Ҫ������  >0.2-<0.2->0.2 �Ĺ���
double blink_EAR_before =0.0;		//գ��ǰ
double blink_EAR_now =1.0;			//գ����
double blink_EAR_after = 0.0;		//գ�ۺ�

Mat Eye_Waveform(420, 420, CV_8UC3, Scalar(255, 255, 255));//���ڼ�¼գ�۵Ĳ���ͼ--��ɫ

double GetEAR(std::vector<full_object_detection> shapes) {
	//��ȡ�����۵�6�����꣨��12���㣩
	//��36������
	unsigned int x_36 = shapes[0].part(36).x();
	unsigned int y_36 = shapes[0].part(36).y();
	//��37������
	unsigned int x_37 = shapes[0].part(37).x();
	unsigned int y_37 = shapes[0].part(37).y();
	//��38������
	unsigned int x_38 = shapes[0].part(38).x();
	unsigned int y_38 = shapes[0].part(38).y();
	//��39������
	unsigned int x_39 = shapes[0].part(39).x();
	unsigned int y_39 = shapes[0].part(39).y();
	//��40������
	unsigned int x_40 = shapes[0].part(40).x();
	unsigned int y_40 = shapes[0].part(40).y();
	//��41������
	unsigned int x_41 = shapes[0].part(41).x();
	unsigned int y_41 = shapes[0].part(41).y();
	//��42������
	unsigned int x_42 = shapes[0].part(42).x();
	unsigned int y_42 = shapes[0].part(42).y();
	//��37������---------------------------------------
	unsigned int x_43 = shapes[0].part(43).x();
	unsigned int y_43 = shapes[0].part(43).y();
	//��38������
	unsigned int x_44 = shapes[0].part(44).x();
	unsigned int y_44 = shapes[0].part(44).y();
	//��39������
	unsigned int x_45 = shapes[0].part(45).x();
	unsigned int y_45 = shapes[0].part(45).y();
	//��40������
	unsigned int x_46 = shapes[0].part(46).x();
	unsigned int y_46 = shapes[0].part(46).y();
	//��41������
	unsigned int x_47 = shapes[0].part(47).x();
	unsigned int y_47 = shapes[0].part(47).y();
	//����EAR
	int left_h1 = y_41 - y_37;			 //37��41���������
	int left_h2 = y_40 - y_38;			 //38��40���������
	double left_h = (left_h1+left_h2) / 2.0;//�۾����¾���
	int left_w = x_39 - x_36;
	if (left_h == 0)left_h = 1;//���۾��պϵ�ʱ�򣬾�����ܼ��Ϊ0����߱ȳ���
	double left_EAR = left_h / left_w;		//�۾���߱�

	int right_h1 = y_47 - y_43;			 //37��41���������
	int right_h2 = y_46 - y_44;			 //38��40���������
	double right_h = (right_h1 + right_h2) / 2.0;//�۾����¾���
	int right_w = x_45 - x_42;
	if (right_h == 0)right_h = 1;//���۾��պϵ�ʱ�򣬾�����ܼ��Ϊ0����߱ȳ���
	double right_EAR = right_h / right_w;		//�۾���߱�

	//ȡ��ֻ�۾���ƽ����߱���Ϊ�۾��Ŀ�߱�
	double EAR = (left_EAR + right_EAR) / 2.0;
	return EAR;
}

void Draw_init() {//����ʼ������
	Point p1 = Point(20, 400);
	Point p2 = Point(20, 0);
	Point p3 = Point(420, 400);
	//����
	cv::line(Eye_Waveform, Point(20, 200), Point(420, 200), Scalar(0, 16, 219), 1, LINE_AA);
	cv::line(Eye_Waveform, p1, p2, Scalar(219, 22, 0), 1, LINE_AA);//��ɫ����
	cv::line(Eye_Waveform, p1, p3, Scalar(219, 22, 0), 1, LINE_AA);
	//����--Ӳдʵ��
	cv::putText(Eye_Waveform, "0", Point(10, 410), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);//Դ��
	cv::putText(Eye_Waveform, "0.1", Point(0, 310), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "0.2", Point(0, 210), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "0.3", Point(0, 110), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "0.4", Point(0, 10), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	//�����
	cv::circle(Eye_Waveform, Point(20, 400), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 300), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 200), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 100), 1, Scalar(0, 0, 0), 2);
	cv::circle(Eye_Waveform, Point(20, 0), 1, Scalar(0, 0, 0), 1);
	//��������
	cv::putText(Eye_Waveform, "Picture1", Point(200, 410), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
	cv::putText(Eye_Waveform, "time", Point(380, 410), FONT_HERSHEY_COMPLEX, 0.4, Scalar(0, 0, 0), 1);
}

void Count(double EAR) {//���������Ŀǰ����ó���EAR
	if (blink_EAR_before < EAR)blink_EAR_before = EAR; 
	if (blink_EAR_now > EAR)blink_EAR_now = EAR; 
	if (blink_EAR_after < EAR)blink_EAR_after = EAR;
	if ((blink_EAR_before > 0.2) && (blink_EAR_now <= 0.2) && (blink_EAR_after > 0.2)) {//���ֵ����������еĺ�������������
		END_TIME = clock();
		cout << "EAR��" << EAR << " ǰ��" << blink_EAR_before << " �У�" << blink_EAR_now << " ��" << blink_EAR_after << endl;
		blink_EAR_before = 0.0;
		blink_EAR_now = 1.0;
		blink_EAR_after = 0.0;
		if ((END_TIME - START_TIME) / CLOCKS_PER_SEC * 1000 > 200)count_blink++;
		START_TIME = clock();
	}
}//�����������һ����ʱ�Ĳ�������

void Draw_now(double EAR) {
	Count(EAR);//˳�����һ��cnt
	eye_now_x = eye_now_x + 1;			//�����꣨ÿ10��������һ���㣩
	eye_now_y = 400*(1-EAR/0.4);		//������
	Point pos1 = Point(eye_pre_x, eye_pre_y);//��һ����
	Point pos2 = Point(eye_now_x, eye_now_y);//���ڵĵ�
	cv::line(Eye_Waveform, pos1, pos2, Scalar(0, 0, 0), 1, LINE_AA);//����-��ɫ
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
	//������ͷ
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "���޷�������ͷ��" << endl;
		return 1;
	}

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pos_model;
	deserialize("E:/CV/shape_predictor_68_face_landmarks.dat")>>pos_model;

	while (true) {
		Mat frame;
		cap >> frame;
		cv_image<bgr_pixel> cimg(frame);//��ͼ��ת��Ϊdlib�е�BGR��ʽ

		std::vector<dlib::rectangle> faces = detector(cimg);
		std::vector<full_object_detection> shapes;
		for (unsigned int i = 0; i < faces.size(); i++)shapes.push_back(pos_model(cimg, faces[i]));

		if (!shapes.empty()) {
			for (int j = 0; j < shapes.size(); j++) {
				for (int i = 0; i < 68; i++) {//����������ֵ�ĵ�
					cv::circle(frame, cvPoint(shapes[j].part(i).x(), shapes[j].part(i).y()), 2, cv::Scalar(219,22,0), -1);
				}
			}
			Draw_now(GetEAR(shapes));//���EAR����Ϊ��������գ�۲���ͼ
			//��hight_left_eye��float����ת�����ַ�������
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