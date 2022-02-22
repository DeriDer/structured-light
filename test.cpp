#include "stdafx.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

class ExtractLines {
private:
		cv::Mat Src,    //Define Mat to hold source image
		Img,    //Original image
		Dst,    //Processed image
		Dx,     //Partial derivative with respect to x
		Dy,     //Partial derivative with respect to y
		Dxx,    //Second derivate with respect to x
		Dxy,    //Second partial derivative with respect to x and y, Dxy = Dyx
		Dyy;    //Second derivative with respect to y
	std::vector<cv::Point> *DetectPoints;
	void ComputeDerivative(void);
	void ComputeHessian(void);
public:
	ExtractLines(const cv::Mat& Img);
	void imshow(void) const;
	inline ~ExtractLines() { delete DetectPoints; }
};

/**
* @brief  Compute first and second partial derivative matrixes.
*         Call ComputeHessian() after all the derivative matrixes
*         has been computed.
**/
void ExtractLines::ComputeDerivative(void) {

	cv::Mat Mask_Dx = (cv::Mat_<float>(3, 1) << 1, 0, -1);
	cv::Mat Mask_Dy = (cv::Mat_<float>(1, 3) << 1, 0, -1);
	cv::Mat Mask_Dxx = (cv::Mat_<float>(3, 1) << 1, -2, 1);
	cv::Mat Mask_Dyy = (cv::Mat_<float>(1, 3) << 1, -2, 1);
	cv::Mat Mask_Dxy = (cv::Mat_<float>(2, 2) << 1, -1, -1, 1);

	cv::filter2D(Dst, Dx, CV_32FC1, Mask_Dx);
	cv::filter2D(Dst, Dy, CV_32FC1, Mask_Dy);
	cv::filter2D(Dst, Dxx, CV_32FC1, Mask_Dxx);
	cv::filter2D(Dst, Dxy, CV_32FC1, Mask_Dxy);
	cv::filter2D(Dst, Dyy, CV_32FC1, Mask_Dyy);

	ComputeHessian();
}

/**
* @brief  Compute first and second partial derivative matrixes.
*         Call ComputeHessian() after all the derivative matrixes
*         has been computed.
**/
void ExtractLines::ComputeHessian(void) {
	std::ofstream points("points.txt", std::ios::out);
	for (int x = 0; x < Dst.cols; x++)
	{
		for (int y = 0; y < Dst.rows; y++)
		{
			if (Src.at<uchar>(y, x) > 10)
			{
				cv::Mat Hessian(2, 2, CV_32FC1), Eigenvalue, Eigenvector;
				Hessian.at<float>(0, 0) = Dxx.at<float>(y, x);
				Hessian.at<float>(0, 1) = Dxy.at<float>(y, x);
				Hessian.at<float>(1, 0) = Dxy.at<float>(y, x);
				Hessian.at<float>(1, 1) = Dyy.at<float>(y, x);

				cv::eigen(Hessian, Eigenvalue, Eigenvector);
				double Nx, Ny;
				if (fabs(Eigenvalue.at<float>(0, 0)) >= fabs(Eigenvalue.at<float>(1, 0)))
				{
					Nx = Eigenvector.at<float>(0, 0);
					Ny = Eigenvector.at<float>(0, 1);
				}
				else
				{
					Nx = Eigenvector.at<float>(1, 0);
					Ny = Eigenvector.at<float>(1, 1);
				}
				double denominator = Dxx.at<float>(y, x) * Nx * Nx + 2 * Dxy.at<float>(y, x) * Nx * Ny + Dyy.at<float>(y, x) * Ny * Ny, t;
				if (denominator != 0.0)
				{
					t = -(Dx.at<float>(y, x) * Nx + Dy.at<float>(y, x) * Ny) / denominator;
					if (fabs(t * Nx) <= 0.5 && fabs(t * Ny) <= 0.5)
					{
						cv::Point_<float> res(x + t * Nx, y + t * Ny);
						if (!points.is_open())
							std::cout << "shit!!!!" << std::endl;
						else
							points << x + t * Nx << ",  " << y + t * Ny << std::endl;
						//std::cout << x + t * Nx << ",  " << y + t * Ny << std::endl;
						(*DetectPoints).push_back(res);
					}
				}
				else
				{
					//points << x << ",  " << y << std::endl;
					//cv::Point_<float> res(x, y);
					//(*DetectPoints).push_back(res);
				}
			}
		}
	}
	points.close();
}

/**
* @brief  Constructor, init the image and preprocess the input image with Gaussian filter.
Throw error if cannot open the image.
@param  img_name: the input file name.
**/
ExtractLines::ExtractLines(const cv::Mat& Input_img) {
	Src = Input_img.clone();
	Img = Input_img.clone();
	//cv::cvtColor(Src, Src, CV_BGR2GRAY);    //Convert RGB image to gray space
	Dst = Src.clone();
	cv::GaussianBlur(Dst, Dst, cv::Size(0, 0), 3, 3);
	DetectPoints = new std::vector<cv::Point>;
	ComputeDerivative();
}

/**
* @brief  Draw detected salient points on original image using cv::circle and display the result.
**/
void ExtractLines::imshow(void) const {
	cv::Mat Img0 = cv::imread("1.BMP", 1);
	for (unsigned int i = 0; i < (*DetectPoints).size(); i++) {
		cv::circle(Img0, (*DetectPoints)[i], 0.3, cv::Scalar(0, 200, 0));
	}
	
	cv::imshow("result", Img0);
	cv::waitKey(0);
}



int main()
{
	cv::Mat img1 = cv::imread("1.BMP", 1);
	cv::Mat img2 = cv::imread("2.BMP", 1);
	cv::Mat channel_1[3];
	cv::Mat channel_2[3];
	cv::split(img1, channel_1);
	cv::split(img2, channel_2);
	cv::Mat diffImage(img1.rows, img1.cols, CV_8UC1);
	cv::Mat test(img1.rows, img1.cols, CV_32FC1);
	cv::Mat dst;
	
	cv::absdiff(channel_1[2], channel_2[2], diffImage);
	cv::absdiff(channel_1[2], channel_2[2], test);
	//diffImage.convertTo(dst, CV_32FC1);
	/*
	for (int i = 0; i < img1.rows; i++)
	{	 
		for (int j = 0; j < img1.cols / 2; j++)
		{
			float* data = test.ptr<float>(i, j);
			//std::cout << *data<< std::endl;
			double minVal, maxVal;
			cv::minMaxLoc(test, &minVal, &maxVal);
			//std::cout << (int) diffImage.at<uchar>(i, j) << std::endl;
			if (1 > 0.5)
			{
				double a =(double) test.at<uchar>(i, j);
				std::cout << "float ??: " << a << std::endl;

			}
		}
	}
	*/
			cv::imshow("diffr", test);
	cv::waitKey(0);
	cv::Mat mask = diffImage.clone();
	std::vector <int> x1;
	std::vector <int> x2;
	for (int i = 0; i < img1.rows; i++)
	{
		int max_x1 = 0;
		int current_intensity = 0;
		for (int j = 0; j < img1.cols/2; j++)
		{
			int r_intensity;
			if (j > 2 && j < img1.cols/2 - 2)
			{
				//r_intensity = diffImage.at<cv::Vec3b>(j, i)[2] + diffImage.at<cv::Vec3b>(j - 1, i)[2] \
					+ diffImage.at<cv::Vec3b>(j + 1, i)[2] + diffImage.at<cv::Vec3b>(j - 2, i)[2] + diffImage.at<cv::Vec3b>(j + 2, i)[2];
				r_intensity = diffImage.at<uint8_t>(i, j - 1) + diffImage.at<uint8_t>(i, j - 2) + diffImage.at<uint8_t>(i,j) + diffImage.at<uint8_t>(i, j + 1) + diffImage.at<uint8_t>(i,j + 2);
			}
			else
				//r_intensity = diffImage.at<cv::Vec3b>(j, i)[2] * 5;
				r_intensity = diffImage.at<uint8_t>(i, j) * 5;
			if (r_intensity > current_intensity)
			{
				max_x1 = j;
				current_intensity = r_intensity;
			}
				
		}
		//std::cout << max_x1 << std::endl;
		x1.push_back(max_x1);

	}
	for (int i = 0; i < img1.rows; i++)
	{
		for (int k = 0; k < img1.cols / 2; k++)
		{
			if (abs(k - x1[i]) > 6)
			{
				diffImage.at<uint8_t>(i, k) = 0;

			}
		}
	}
	
	for (int i = 0; i < img1.rows; i++)
	{
		int max_x2 = 0;
		int current_intensity = 0;
		for (int j = img1.cols / 2; j < img1.cols; j++)
		{
			int r_intensity;
			if (j > img1.cols / 2 + 2 && j < img1.cols - 2)
			{
				r_intensity = diffImage.at<uint8_t>(i, j - 1) + diffImage.at<uint8_t>(i, j - 2) + diffImage.at<uint8_t>(i, j) + diffImage.at<uint8_t>(i, j + 1) + diffImage.at<uint8_t>(i, j + 2);
			}
			else
				r_intensity = diffImage.at<uint8_t>(i, j) * 5;
			if (r_intensity > current_intensity)
			{
				max_x2 = j;
				current_intensity = r_intensity;
			}

		}
		x2.push_back(max_x2);
	}
	
	for (int i = 0; i < img1.rows; i++)
	{
		for (int k = img1.cols / 2; k < img1.cols; k++)
		{
			if (abs(k - x2[i]) > 6)
			{
				diffImage.at<uint8_t>(i, k) = 0;

			}
		}
	}

	cv::Mat Src = diffImage;
	cv::imwrite("mask.BMP", Src);
	if (!Src.data) {
		printf("fail to open the image!\n");
		system("pause");
		return -1;
	}
		
	
	//sobel 1st
	cv::Mat src_gray = Src.clone();
	cv::Mat first;
	src_gray.convertTo(Src, CV_32FC1);
	first.convertTo(first, CV_32FC1);
	cv::Sobel(src_gray, first, CV_32F,1, 0, 3);
	cv::Mat second;
	second.convertTo(second, CV_32FC1);
	cv::Sobel(first, second, CV_32F, 1, 0, 3);
	//cv::imshow("first", first);
	std::cout << first.at<int>(100, 200) << std::endl;
	for (int i = 948; i < Src.rows/*Src.rows*/; i++)
	{
		std::cout << x1[i] << std::endl;
		std::cout << "i = "<< i << std::endl;
		for (int j = x1[i] - 5; j < x1[i] + 5; j++)
		{
			//std::cout << "1: " << first.at<float>(i, j)  << " 2: " << second.at<float>(i, j) << std::endl;
			if (first.at<float>(i,j) >0 && first.at<float>(i, j+1) < 0 && abs(first.at<float>(i, j)-first.at<float>(i, j + 1))>200 )
			{
				
				std::cout << " x:  "<< j+1+(first.at<float>(i, j)/second.at<float>(i,j)) << std::endl/*(float) first.at<float>(i, j)*/;
			}
			
		}
		std::cout << std::endl;
	}
	system("pause");
	/*/algorithm
	ExtractLines lines(Src);
	lines.imshow();
	cv::waitKey(0);
	
	std::cout << "sth happene" << std::endl;
	system("pause");
	*/
	return 0;

}


