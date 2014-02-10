#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;
//Parameter
	//Camera setting
	int camera=1;
	//Inrange setting
	Scalar YHthres(20,255,255);
	Scalar YLthres(0,110,0);
	Scalar GHthres(70,230,255);
	Scalar GLthres(55,150,0);
	Scalar BHthres(100,230,255);
	Scalar BLthres(80,150,0);
	//Erode Dilate setting 
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	//contour Area setting
	double Min_contour_area=1000;


//Variable
vector<Point>  Ycontours;
vector<Point>  Gcontours;
vector<Point>  Bcontours;
vector<Vec4i> hierarchy;
struct tri_cal_para {
	double diff_dis_x;
	double diff_dis_y;
	double angle;
};
vector<tri_cal_para> result;
Mat dst_norm,dst_norm_scaled ;


//User defined Functions
Mat erode_dilate(Mat img){
	erode(img,img, element);
	erode(img,img, element);
	dilate(img,img, element );

	return img;
}

double length(Point a,Point b){
	double diff_x=abs(a.x-b.x);
	double diff_y=abs(a.y-b.y);
	return sqrt(diff_x*diff_x+diff_y*diff_y);
}

tri_cal_para tri_cal(vector<vector<Point>> contours);
vector<vector<Point>> contours_tri_filter(vector<vector<Point>> contours);
void frame_process_result_pushback(Mat frame,double high_thres);
tri_cal_para mean_filter(vector<tri_cal_para> result);


//Main function
int main(){
	VideoCapture cap(camera);
	Mat frame,Grayframe,trash,Cframe,HSVframe,Yframe,Gframe,Bframe;
	double high_thres;
	for(;;){
		cap>>frame;
		cvtColor(frame,Grayframe,CV_BGR2GRAY);
		high_thres=threshold(Grayframe,trash,0,255,CV_THRESH_BINARY+CV_THRESH_OTSU);
		Canny( frame, Cframe, high_thres*0.5, high_thres);
		
		cvtColor(frame,HSVframe,CV_BGR2HSV);
		inRange(HSVframe,YLthres,YHthres,Yframe);
		inRange(HSVframe,GLthres,GHthres,Gframe);
		inRange(HSVframe,BLthres,BHthres,Bframe);

		frame_process_result_pushback(Yframe,high_thres);
		frame_process_result_pushback(Gframe,high_thres);
		frame_process_result_pushback(Bframe,high_thres);

		//cout<<"Result Size: "<<result.size()<<endl;
		for (int i=0;i<result.size();i++){
			cout<<i<<"--"<<result[i].angle*180/CV_PI<<"--"<<result[i].diff_dis_x<<"--"<<result[i].diff_dis_y<<endl;
		}

		tri_cal_para solution=mean_filter(result);

		if(!(solution.angle==0&&solution.diff_dis_x==0&&solution.diff_dis_y==0)){
			cout<<"Angle: "<<solution.angle*180/CV_PI;
			cout<<"          ";
			cout<<"Dis_X: "<<solution.diff_dis_x;
			cout<<"          ";
			cout<<"Dis_Y: "<<solution.diff_dis_y<<endl;
		}
		//for( int i = 0; i< contours_poly.size(); i++ )
		//{
		//	drawContours( frame, contours_poly, i, Scalar(255,0,0), 1, 8, vector<Vec4i>(), 0, Point() );	
		//}

		for(int i=0;i<result.size();i++){
			result.erase(result.begin()+i);
			i--;
		}

		imshow("Canny",Cframe);
		imshow("webcam",frame);
		if (waitKey(10)==32) break;
	}
}

//User define function
vector<vector<Point>> contours_tri_filter(vector<vector<Point>> contours){
	vector<vector<Point> > contours_poly( contours.size() );
	for( int i = 0; i< contours.size(); i++ )
		approxPolyDP( Mat(contours[i]), contours_poly[i],5, true );
	
	for( int i = 0; i< contours_poly.size(); i++ )
		if (contours_poly[i].size()!=3||contourArea(contours_poly[i])<Min_contour_area){
			contours_poly.erase(contours_poly.begin()+i);
			i--;
		}
	
	if (contours_poly.size()>=2)
		//Do somethings to reduce contours size to 1
		for(int i=1 ;i< contours_poly.size();i++){
			contours_poly.erase(contours_poly.begin()+i);
			i--;
		}
	return contours_poly;
}


tri_cal_para tri_cal(vector<vector<Point>> contours_poly,Size frame_size){
	tri_cal_para ans;
	Point triangle[3];
		if (contours_poly.size()==1) {
			//figure out the pointing direction
			double len1=length(contours_poly[0][0],contours_poly[0][1]);
			double len2=length(contours_poly[0][1],contours_poly[0][2]);
			double len3=length(contours_poly[0][0],contours_poly[0][2]);
			Point MD(frame_size.width/2,frame_size.height/2);
			bool find_tri=false;
			if (len1<=len2 && len1<=len3){
				triangle[0]=contours_poly[0][2];
				triangle[1]=contours_poly[0][0];
				triangle[2]=contours_poly[0][1];
				find_tri=true;
			}
			if (len2<=len3 && len2<=len1){
				triangle[0]=contours_poly[0][0];
				triangle[1]=contours_poly[0][1];
				triangle[2]=contours_poly[0][2];
				find_tri=true;
			}
			if (len3<=len2 && len3<= len1){
				triangle[0]=contours_poly[0][1];
				triangle[1]=contours_poly[0][2];
				triangle[2]=contours_poly[0][0];
				find_tri=true;
			}
			if (find_tri==true){
			Point Zero(0,0);
			Point M_B((triangle[1].x+triangle[2].x)/2,(triangle[1].y+triangle[2].y)/2);
			double tri_height=length(M_B,triangle[0]);
			ans.angle=-atan2((double)M_B.x-triangle[0].x,(double)M_B.y-triangle[0].y);
			double moved_dis=length(MD,triangle[0])-tri_height/2;
			Point mid_tri((M_B.x+triangle[0].x)/2,(M_B.y+triangle[0].y)/2);
			ans.diff_dis_x=MD.x-mid_tri.x;
			ans.diff_dis_y=MD.y-mid_tri.y;
			}
		}
	return ans;
}

void frame_process_result_pushback(Mat frame,double high_thres){
		frame=erode_dilate(frame);
		//Yframe process
		Canny( frame, frame, high_thres*0.5, high_thres);
		vector<vector<Point>> contours;
		vector<vector<Point>> Poly_contours;
		findContours( frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		Poly_contours=contours_tri_filter(contours);
		//cout<<"Contours size: "<<Poly_contours.size()<<endl;
		if (Poly_contours.size()==1){
			result.push_back(tri_cal(Poly_contours,frame.size()));
		}
}

tri_cal_para mean_filter(vector<tri_cal_para> result){
	int SIZE=result.size();
	tri_cal_para ans;
	ans.angle=0;
	ans.diff_dis_x=0;
	ans.diff_dis_y=0;
	if (SIZE>0){
		for(int i=0;i<SIZE;i++){
			ans.diff_dis_x+=result[i].diff_dis_x;
			ans.angle+=result[i].angle;
			ans.diff_dis_y+=result[i].diff_dis_y;
		}
		ans.angle/=SIZE;
		ans.diff_dis_x/=SIZE;
		ans.diff_dis_y/=SIZE;
	}
	return ans;
}