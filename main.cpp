#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void DFT(Mat image, Mat &image_dft_complex,Mat &image_dft);
void IDFT(Mat image_dft_complex,Mat &image_idft);
void line_erase(Mat image_src,Mat &image_dst,int winsize);
void line_erase2(Mat image_src,Mat &image_dst,int winsize);
void my_threshold(Mat image_src,Mat &image_dst,float value1);
void my_threshold2(Mat image_src,Mat &image_dst,int value1);
Mat filter(Mat padded);
void get_works(Mat image_origin,Mat refer,Mat &image_dst);
void cover(Mat origin,Mat &src,Mat refer);
Mat extract_frequency(Mat image_src);
Mat extract_normal(Mat image_src);
void area_grow(Mat image_src, Mat &image_res,Point point);
void get_element(Mat image_origin,Mat refer,Mat &image_dst);
void split(Mat image_origin,Mat src,Mat &image1,Mat &image2,Mat &image3);

int main()
{
	Mat image_origin = imread("HW2.jpg",0);
	int flag=1;
	Mat image_end;
	Mat element1,element2,element3;
	while(flag)
	{
		cout<<"��ѡ��Ƶ��ȥ�뻹�ǿ���ȥ�룺"<<endl<<"0.�˳�"<<endl<<"1.Ƶ��"<<endl<<"2.����"<<endl;
		cin>>flag;
		switch(flag)
		{
		case 1:
			image_end = extract_frequency(image_origin);
			imshow("���ͼ",image_end);
			imwrite("Ƶ����.png",image_end);
			split(image_origin,image_end,element1,element2,element3);
			imshow("Ԫ��1",element1);
	        imshow("Ԫ��2",element2);
	        imshow("Ԫ��3",element3);
			imwrite("Ƶ��_Ԫ��1.png",element1);
			imwrite("Ƶ��_Ԫ��2.png",element2);			
			imwrite("Ƶ��_Ԫ��3.png",element3);
			waitKey();
	        destroyAllWindows();
			break;
		case 2:
			image_end = extract_normal(image_origin);
			imshow("���ͼ",image_end);			
			imwrite("������.png",image_end);
			split(image_origin,image_end,element1,element2,element3);
			imshow("Ԫ��1",element1);
	        imshow("Ԫ��2",element2);
	        imshow("Ԫ��3",element3);			
			imwrite("����_Ԫ��1.png",element1);
			imwrite("����_Ԫ��2.png",element2);			
			imwrite("����_Ԫ��3.png",element3);
			waitKey();
	        destroyAllWindows();
			break;
		default:
			break;
		}
	}
	return 0;
}
Mat extract_frequency(Mat image_src)
{
	Mat image_origin = image_src.clone();
	Mat image_threshold = image_src.clone();
	my_threshold2(image_origin,image_threshold,(int)65);

	Mat image_dft_Img;
	Mat image_dft;
	DFT(image_origin,image_dft_Img,image_dft);

	Mat image_dft_Img_grid = filter(image_dft_Img);

	Mat image_idft_grid;
	Mat image_origin_normal;//��С�Ѿ���չ��0-1
	IDFT(image_dft_Img_grid,image_idft_grid);
	IDFT(image_dft_Img,image_origin_normal);

	Mat image_idft_element;
	image_idft_element = image_origin_normal - 0.9*image_idft_grid;

	Mat image_idft_element_threshold = image_idft_element.clone();
	my_threshold(image_idft_element,image_idft_element_threshold,(float)0.22);

	Mat image_covered;//��Сδ��չ��0-255
	get_works(image_origin,image_idft_element_threshold,image_covered);
	//imshow("��ֵ������",image_covered);

	Mat image_open = image_threshold.clone();//δ��չ��0-1
	morphologyEx(image_open, image_open, MORPH_OPEN, Mat(12, 12, CV_32FC1));//������
	//imshow("��������",image_open);

	cover(image_origin,image_covered,image_open);
	//imshow("����������",image_covered);

	line_erase(image_covered,image_covered,11);
	line_erase(image_covered,image_covered,11);

	return image_covered;
}
Mat extract_normal(Mat image_src)
{
	Mat image_origin = image_src.clone();
	Mat image_threshold = image_src.clone();
	my_threshold2(image_origin,image_threshold,(int)65);
	for(int i  =0;i<5;i++)
	{
		line_erase(image_threshold,image_threshold,25);
	}
	for(int i =0;i <= 40;i++)
	{
		line_erase2(image_threshold,image_threshold,9);
	}
	return image_threshold;
}
void split(Mat image_origin,Mat src,Mat &canny1,Mat &canny2,Mat &canny3)
{
	Mat image_canny = src.clone();
	Canny(src,src,60,15);
	Canny(image_canny,image_canny,300,100);

	morphologyEx(src, src, MORPH_CLOSE, Mat(4, 4, CV_32FC1));
	morphologyEx(image_canny, image_canny, MORPH_CLOSE, Mat(4, 4, CV_32FC1));

	Point seed1(255,255);
	Point seed3(240,400);

	canny3 = Mat::zeros(image_canny.size(),CV_8UC1);
	area_grow(image_canny,canny3,seed3);
	canny1 = Mat::zeros(image_canny.size(),CV_8UC1);
	area_grow(image_canny,canny1,seed1);
	canny2 = Mat::zeros(src.size(),CV_8UC1);	
	area_grow(src,canny2,seed1);
	canny1 = canny1 - canny2;

	line_erase(canny1,canny1,15);
	get_element(image_origin,canny1,canny1);
	get_element(image_origin,canny2,canny2);
	get_element(image_origin,canny3,canny3);
}
void get_element(Mat image_origin,Mat refer,Mat &image_dst)
{
	image_dst=image_origin.clone();
	for(int i = 0;i<image_origin.rows;i++)
	{
		for(int j= 0;j<image_origin.cols;j++)
		{
			if(refer.ptr<uchar>(i)[j] > 50)
			{
				image_dst.ptr<uchar>(i)[j] = image_origin.ptr<uchar>(i)[j];
			}
			else
			{
				image_dst.ptr<uchar>(i)[j] = 0;
			}
		}
	}
}
void DFT(Mat image, Mat &image_dft_complex,Mat &image_dft)
{
	
	int M = getOptimalDFTSize( image.rows );
    int N = getOptimalDFTSize( image.cols );
    Mat padded;
	//��ԭͼ��Ĵ�С��Ϊm*n�Ĵ�С�������λ����0��
    copyMakeBorder(image, padded, 0, M - image.rows, 0, N - image.cols, BORDER_CONSTANT, Scalar::all(0));
	//�����ǻ�ȡ������mat��һ�����ڴ��dft�任��ʵ����һ�����ڴ���鲿����ʼ��ʱ��ʵ������ͼ�����鲿ȫΪ0
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
	//��������ͨ����mat�ںϳ�һ����ͨ����mat�������ںϵ�complexImg����ʵ���������鲿
    merge(planes, 2, complexImg);
	//dft�任����ΪcomplexImg�����������ͨ����mat������dft�任�Ľ��Ҳ���Ա���������
    dft(complexImg, complexImg); 
	//idft(complexImg,complexImg);
	//��complexImg���²�ֳ�����mat��һ����ʵ����һ�����鲿
    split(complexImg, planes);

    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    //��һ������Ϊ�˼���dft�任��ķ�ֵ���Ա�����ʾ��ֵ�ļ��㹫ʽ����
	image_dft_complex = complexImg.clone();
    magnitude(planes[0], planes[1], planes[0]);//������mat��Ӧλ�����
    image_dft = planes[0];
    image_dft += Scalar::all(1);
    log(image_dft, image_dft);

    //�޼�Ƶ�ף����ͼ����л������������Ļ�������Ƶ���ǲ��ԳƵģ����Ҫ�޼�
    image_dft = image_dft(Rect(0, 0, image_dft.cols & -2, image_dft.rows & -2));
	image_dft_complex= image_dft_complex(Rect(0, 0, image_dft_complex.cols & -2, image_dft_complex.rows & -2));
	//image_dft_i = image_dft_i(Rect(0, 0, image_dft_i.cols & -2, image_dft_i.rows & -2));
    int cx = image_dft.cols/2;
    int cy = image_dft.rows/2;
	
    Mat tmp;
    Mat q0(image_dft, Rect(0, 0, cx, cy));
    Mat q1(image_dft, Rect(cx, 0, cx, cy));
    Mat q2(image_dft, Rect(0, cy, cx, cy));
    Mat q3(image_dft, Rect(cx, cy, cx, cy));
	
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
	
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

	Mat q00(image_dft_complex, Rect(0, 0, cx, cy));
    Mat q11(image_dft_complex, Rect(cx, 0, cx, cy));
    Mat q22(image_dft_complex, Rect(0, cy, cx, cy));
    Mat q33(image_dft_complex, Rect(cx, cy, cx, cy));
	
    q00.copyTo(tmp);
    q33.copyTo(q00);
    tmp.copyTo(q33);
	
    q11.copyTo(tmp);
    q22.copyTo(q11);
    tmp.copyTo(q22);

    normalize(image_dft, image_dft, 0, 1, CV_MINMAX);
}
void IDFT(Mat complexImg,Mat &image_idft)
{
	//Mat planes[] = {image_dft_r,image_dft_i};
	Mat planes[] = {Mat::zeros(complexImg.size(),CV_32F), Mat::zeros(complexImg.size(), CV_32F)};
	int cx = complexImg.cols/2;
    int cy = complexImg.rows/2;
	
    Mat tmp;
    Mat q0(complexImg, Rect(0, 0, cx, cy));
    Mat q1(complexImg, Rect(cx, 0, cx, cy));
    Mat q2(complexImg, Rect(0, cy, cx, cy));
    Mat q3(complexImg, Rect(cx, cy, cx, cy));
	
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
	
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
	
	
	idft(complexImg,complexImg);	
	split(complexImg, planes);
	magnitude(planes[0], planes[1], planes[0]);
	image_idft = planes[0].clone();
	normalize(image_idft, image_idft, 0, 1, CV_MINMAX);
}
void line_erase(Mat image_src,Mat &image_dst,int winsize)
{
	int num;
	copyMakeBorder(image_src, image_src, winsize/2, winsize/2, winsize/2, winsize/2, BORDER_CONSTANT, Scalar::all(0));
	for(int i=0;i<image_dst.rows;i++)
	{
		for(int j=0;j<image_dst.cols;j++)
		{
			num = 0;
			for(int k = -winsize/2;k <= winsize/2;k++)
			{
				for(int l = -winsize/2;l <= winsize/2;l++)
				{
					if(image_src.ptr<uchar>(i + winsize/2 + k)[j + winsize/2 + l] >= 65)
					{
						num = num +1;
					}
				}
			}
			if(num <= winsize*winsize*7/20)
			{
				image_dst.ptr<uchar>(i)[j] = 0;
			}
		}
	}
}
void line_erase2(Mat image_src,Mat &image_dst,int winsize)
{
	int num;
	copyMakeBorder(image_src, image_src, winsize/2, winsize/2, winsize/2, winsize/2, BORDER_CONSTANT, Scalar::all(0));
	for(int i=0;i<image_dst.rows;i++)
	{
		for(int j=0;j<image_dst.cols;j++)
		{
			num = 0;
			for(int k = -winsize/2;k <= winsize/2;k++)
			{
				for(int l = -winsize/2;l <= winsize/2;l++)
				{
					if(image_src.ptr<uchar>(i + winsize/2 + k)[j + winsize/2 + l] >= 65)
					{
						num = num +1;
					}
				}
			}
			if(num <= winsize*winsize*45/100)
			{
				image_dst.ptr<uchar>(i)[j] = 0;
			}
		}
	}
}
void my_threshold(Mat image_src,Mat &image_dst,float value1)
{
	for(int i = 0;i<image_dst.rows;i++)
	{
		for(int j = 0;j<image_dst.cols;j++)
		{
			if(image_src.at<float>(i,j) < value1)
			{
				image_dst.at<float>(i,j) = 0;
			}
		}
	}
}
void my_threshold2(Mat image_src,Mat &image_dst,int value1)
{
	for(int i = 0;i<image_dst.rows;i++)
	{
		for(int j = 0;j<image_dst.cols;j++)
		{
			if(image_src.at<uchar>(i,j) < value1)
			{
				image_dst.at<uchar>(i,j) = 0;
			}
		}
	}
}
Mat filter(Mat image_src)
{
	Mat temp = image_src.clone();
	Mat planes[] = {Mat::zeros(temp.size(),CV_32F), Mat::zeros(temp.size(), CV_32F)};
	split(temp, planes);

	Mat gaussian_blur_1(temp.size(), CV_32FC1);
	Mat gaussian_blur_2(temp.size(), CV_32FC1);
	Mat gaussian_blur_low(temp.size(), CV_32FC1);
	Mat gaussian_blur_final(temp.size(), CV_32FC1);
	//���ø�˹������������͵�Ƶ�˲���
	for (int i = 0; i<temp.rows; i++)
	{
		float*p = gaussian_blur_1.ptr<float>(i);
		for (int j = 0; j<temp.cols; j++)
		{
			float d1 = pow(j - temp.cols / 2, 2) + 1;
			float x;
			x = abs(i - temp.rows / 2) + 100.0;
			p[j] = float(expf(-d1 / (4 * x)));
		}
	}

	for (int i = 0; i<temp.rows; i++)
	{
		float*p = gaussian_blur_2.ptr<float>(i);
		for (int j = 0; j<temp.cols; j++)
		{
			float d2 = pow(i - temp.rows / 2, 2) + 1;
			float x;
			x = abs(j - temp.cols / 2) + 45.0;
			p[j] = float(expf(-d2 / (4 * x)));
		}
	}

	for (int i = 0; i<temp.rows; i++)
	{
		float*p = gaussian_blur_low.ptr<float>(i);
		for (int j = 0; j<temp.cols; j++)
		{
			float d1 = pow(j - temp.cols / 2, 2);
			float d2 = pow(i - temp.rows / 2, 2);
			p[j] = float(expf(-d1 / 150.0))*float(expf(-d2 / 150.0));
		}
	}
	//�ϳ������˲���
	for (int i = 0; i<temp.rows; i++)
	{
		float*p = gaussian_blur_final.ptr<float>(i);
		float*q = gaussian_blur_1.ptr<float>(i);
		float*r = gaussian_blur_2.ptr<float>(i);
		for (int j = 0; j<temp.cols; j++)
		{

			if (((3 * j) >= (4 * i)) && ((3 * j) < (4 * (temp.rows - i))))
			{
				p[j] = q[j];
			}
			if ((3 * j) < (4 * i) && (3 * j) >= (4 * (temp.rows - i)))
			{
				p[j] = q[j];
			}
			if ((3 * j) >= (4 * i) && (3 * j) >= (4 * (temp.rows - i)))
			{
				p[j] = r[j];
			}
			if ((3 * j) <(4 * i) && (3 * j)<(4 * (temp.rows - i)))
			{
				p[j] = r[j];
			}
		}
	}
	for (int i = 0; i<temp.rows; i++)
	{
		for (int j = 0; j<temp.cols; j++)
		{
			planes[0].ptr<float>(i)[j] = planes[0].ptr<float>(i)[j] * (1-gaussian_blur_low.ptr<float>(i)[j]) * gaussian_blur_final.ptr<float>(i)[j];
			planes[1].ptr<float>(i)[j] = planes[1].ptr<float>(i)[j] * (1-gaussian_blur_low.ptr<float>(i)[j]) * gaussian_blur_final.ptr<float>(i)[j];
			
	}
	merge(planes, 2, temp);
	return temp;
	}
}
void get_works(Mat image_origin,Mat refer,Mat &image_dst)
{
	image_dst=image_origin.clone();
	for(int i = 0;i<image_origin.rows;i++)
	{
		for(int j= 0;j<image_origin.cols;j++)
		{
			if(refer.ptr<float>(i)[j] > 0.22)
			{
				image_dst.ptr<uchar>(i)[j] = image_origin.ptr<uchar>(i)[j];
			}
			else
			{
				image_dst.ptr<uchar>(i)[j] = 0;
			}
		}
	}
}
void cover(Mat origin,Mat &src,Mat refer)
{
	for(int i = 0;i<refer.rows;i++)
	{
		for(int j= 0;j<refer.cols;j++)
		{
			if(refer.ptr<uchar>(i)[j] > 0)
			{
				src.ptr<uchar>(i)[j] = origin.ptr<uchar>(i)[j];
			}
		}
	}
}
void area_grow(Mat image_src, Mat &image_res,Point point)
{
	vector<Point> seeds;
	Point seed;
	Mat flag = Mat::zeros(image_src.size(),CV_8UC1);

	Point connects[4] = {  Point(0, -1), Point(1, 0),  Point(0, 1), Point(-1, 0)};
	seeds.push_back(point);

	while(seeds.size() != 0)
	{
		seed = seeds.back();
		seeds.pop_back();
		flag.ptr<uchar>(seed.x)[seed.y] = 1;
		image_res.ptr<uchar>(seed.x)[seed.y] = 150;

		for (int i = 0; i < 4; i++)
		{
			int tmpx = seed.x + connects[i].x;
			int tmpy = seed.y + connects[i].y;

			if (tmpx >=1 && tmpy >= 1 && tmpx < image_src.rows && tmpy < image_src.cols)
			{
			   if(!flag.ptr<uchar>(tmpx)[tmpy])
			   {
			      int value = image_src.ptr<uchar>(tmpx)[tmpy] -image_src.ptr<uchar>(seed.x)[seed.y];
			      if(value==0)
			      {
			   	     seeds.push_back(Point(tmpx,tmpy));
			      }
			   }
			}
		}
	}
}
