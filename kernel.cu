#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <windows.h>
#include <gdiplus.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace Gdiplus;
using namespace cv;
using namespace cv::utils;

__device__
void cudaKernelSum(unsigned char* image, int rows, int cols, int channels, int step, int x, int y, int size, int* sum) {
	int numPixels = 0;

	for (int i = (x - (size / 2)); i < (x + (size / 2)) + 1; i++) {
		for (int j = (y - (size / 2)); j < (y + (size / 2)) + 1; j++) {
			if (i >= 0 && j >= 0 && i < cols && j < rows) {
				sum[0] += image[i * channels + j * step];
				sum[1] += image[i * channels + j * step + 1];
				sum[2] += image[i * channels + j * step + 2];
				numPixels++;
			}
		}
	}
	sum[0] = sum[0] / numPixels;
	sum[1] = sum[1] / numPixels;
	sum[2] = sum[2] / numPixels;

}


__global__ void blur_image_kernel(unsigned char* image, unsigned char* blurImage, int rows, int cols, int channels, int step, int size) {

	int index = threadIdx.x + (blockDim.x * blockIdx.x);
	int numPixels = 0;
	int sum[3] = { 0,0,0 };

	int a = index % cols;
	int b = index / cols;

	for (int i = (a - (size / 2)); i < (a + (size / 2)) + 1; i++) {
		for (int j = (b - (size / 2)); j < (b + (size / 2)) + 1; j++) {
			if (i >= 0 && j >= 0 && i < cols && j < rows) {
				sum[0] += image[i * channels + j * step];
				sum[1] += image[i * channels + j * step + 1];
				sum[2] += image[i * channels + j * step + 2];
				numPixels++;
			}
		}
	}
	sum[0] = sum[0] / numPixels;
	sum[1] = sum[1] / numPixels;
	sum[2] = sum[2] / numPixels;


	blurImage[channels * a + step * b] = sum[0];
	blurImage[channels * a + step * b + 1] = sum[1];
	blurImage[channels * a + step * b + 2] = sum[2];
}

unsigned char* blur_image_cuda(unsigned char* image, int rows, int cols, int channels, int step, int size) {

	int threadsPerBlock = 1024;
	int numBlocks = ((rows * cols) / 1024) + 1;

	unsigned char* cudaImage;
	unsigned char* cudaBlurImage;
	cudaMallocManaged(&cudaImage, sizeof(unsigned char) * rows * cols * channels);
	cudaMallocManaged(&cudaBlurImage, sizeof(unsigned char) * rows * cols * channels);

	memcpy(cudaImage, image, sizeof(unsigned char) * rows * cols * channels);
	memset(cudaBlurImage, 0, sizeof(unsigned char) * rows * cols * channels);

	blur_image_kernel << <numBlocks, threadsPerBlock >> > (cudaImage, cudaBlurImage, rows, cols, channels, step, size);
	cudaDeviceSynchronize();

	unsigned char* blurImage = (unsigned char*)malloc(sizeof(unsigned char) * rows * cols * channels);
	memcpy(blurImage, cudaBlurImage, sizeof(unsigned char) * rows * cols * channels);

	cudaFree(cudaImage);
	cudaFree(cudaBlurImage);
	return blurImage;

}

__global__ void convert_greyscale_kernel(unsigned char* image, unsigned char* grayImage, int rows, int cols, int channels, int step) {

	int index = threadIdx.x + (blockDim.x * blockIdx.x);

	int y = index / cols;
	int x = index % cols;

	int blue = (int)image[channels * x + step * y];
	int green = (int)image[channels * x + step * y + 1];
	int red = (int)image[channels * x + step * y + 2];

	grayImage[x + cols * y] = (unsigned char)(.3 * red) + (.59 * green) + (.11 * blue);
}

unsigned char* convert_greyscale_cuda(unsigned char* image, int rows, int cols, int channels, int step) {

	int threadsPerBlock = 1024;
	int numBlocks = ((rows * cols) / 1024) + 1;

	unsigned char* cudaImage;
	unsigned char* cudaGrayImage;	
	cudaMallocManaged(&cudaImage, sizeof(unsigned char) * rows * cols * channels);
	cudaMallocManaged(&cudaGrayImage, sizeof(unsigned char) * rows * cols);

	memcpy(cudaImage, image, sizeof(unsigned char) * rows * cols * channels);
	memset(cudaGrayImage, 0, sizeof(unsigned char) * rows * cols);

	convert_greyscale_kernel << <numBlocks, threadsPerBlock >> > (cudaImage, cudaGrayImage, rows, cols, channels, step);
	cudaDeviceSynchronize();

	unsigned char* grayImage = (unsigned char*)malloc(sizeof(unsigned char) * rows * cols);
	memcpy(grayImage, cudaGrayImage, sizeof(unsigned char) * rows * cols);

	cudaFree(cudaImage);
	cudaFree(cudaGrayImage);
	return grayImage;

}

uchar* blur_image_cpu(uchar* image, int rows, int cols, int channels, int step, int size) {

	uchar* blurImage = (uchar*)malloc(sizeof(uchar) * rows * cols * channels);
	memset(blurImage, 0, sizeof(uchar) * rows * cols * channels);


	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			//int* average = kernelSum(image, rows, cols, channels, step, x, y, size);
			int* sum = (int*)malloc(3 * sizeof(int));
			memset(sum, 0, 3 * sizeof(int));
			int numPixels = 0;
			for (int i = (x - (size / 2)); i < (x + (size / 2)) + 1; i++) {
				for (int j = (y - (size / 2)); j < (y + (size / 2)) + 1; j++) {
					if (i >= 0 && j >= 0 && i < cols && j < rows) {
						sum[0] += image[i * channels + y * step];
						sum[1] += image[i * channels + y * step + 1];
						sum[2] += image[i * channels + y * step + 2];
						numPixels++;
					}
				}
			}
			sum[0] = sum[0] / numPixels;
			sum[1] = sum[1] / numPixels;
			sum[2] = sum[2] / numPixels;

			blurImage[channels * x + step * y] = sum[0];
			blurImage[channels * x + step * y + 1] = sum[1];
			blurImage[channels * x + step * y + 2] = sum[2];
		}
	}
	return blurImage;
}

uchar* grayscale_image_cpu(uchar* image, int rows, int cols, int channels, int step) {

	uchar* grayImage = (uchar*)malloc(sizeof(uchar) * rows * cols);
	memset(grayImage, 0, sizeof(uchar) * rows * cols);

	for (int y = 0; y < rows; y++) {
		for (int x = 0; x < cols; x++) {
			int blue = (int)image[channels * x + step * y];
			int green = (int)image[channels * x + step * y + 1];
			int red = (int)image[channels * x + step * y + 2];

			grayImage[x + cols * y] = (uchar)(.3 * red) + (.59 * green) + (.11 * blue);

		}
	}
	return grayImage;
}


int main(int argc, char** argv) {


	Mat image;
	image = imread("ship-img.jpg", CV_LOAD_IMAGE_COLOR);
	//image = imread("range-rover .jpg", CV_LOAD_IMAGE_COLOR); 
	namedWindow("Display Window", WINDOW_AUTOSIZE);
	imshow("Display Window", image);

	//----------------------------------- CUDA Blur Image --------------------------------------------//

	clock_t start_gpu_BI, end_gpu_BI;
	float total_time_BI;
	start_gpu_BI = clock();

	unsigned char* blurImageData = blur_image_cuda(image.data, image.rows, image.cols, image.channels(), image.step, 6);

	end_gpu_BI = clock();
	//time count stops 
	total_time_BI = ((float)(end_gpu_BI - start_gpu_BI)) / CLOCKS_PER_SEC;
	//calulate total time
	printf("\nTime taken for blurring image in GPU: %f \n", total_time_BI);

	Mat processedImage = Mat(image.rows, image.cols, CV_8UC3, blurImageData);

	//Window name and size
	namedWindow("Display GPU Blur Image", WINDOW_AUTOSIZE);
	//Display blurr Image
	imshow("Display GPU Blur Image", processedImage);

	//----------------------------------- CUDA Gray Scale --------------------------------------------//

	clock_t start_gpu_GS, end_gpu_GS;
	float total_time_GS;
	start_gpu_GS = clock();

	unsigned char* grayImageData = convert_greyscale_cuda(image.data, image.rows, image.cols, image.channels(), image.step);

	end_gpu_GS = clock();
	//time count stops 
	total_time_GS = ((float)(end_gpu_GS - start_gpu_GS)) / CLOCKS_PER_SEC;
	//calulate total time
	printf("\nTime taken for converting image to greyscale in GPU: %f \n", total_time_GS);


	Mat processed_grey_Image = Mat(image.rows, image.cols, CV_8UC1, grayImageData);

	//Window name and size
	namedWindow("Display GPU Grey Image", WINDOW_AUTOSIZE);
	// Display grey scale Image
	imshow("Display GPU Grey Image", processed_grey_Image);


	//----------------------------------- CPU Blur Image --------------------------------------------//

	clock_t start_cpu_BI, end_cpu_BI;
	float total_time_cpu_BI;
	start_cpu_BI = clock();

	unsigned char* blurImageData_cpu = blur_image_cpu(image.data, image.rows, image.cols, image.channels(), image.step, 6);

	end_cpu_BI = clock();
	//time count stops 
	total_time_cpu_BI = ((float)(end_cpu_BI - start_cpu_BI)) / CLOCKS_PER_SEC;
	//calulate total time
	printf("\nTime taken for blurring image in CPU: %f \n", total_time_cpu_BI);


	Mat processedImage_cpu = Mat(image.rows, image.cols, CV_8UC3, blurImageData_cpu);
	//Window name and size
	namedWindow("Display CPU Blur Image", WINDOW_AUTOSIZE);
	// Display grey scale Image
	imshow("Display CPU Blur Image", processedImage_cpu);


	//----------------------------------- CPU Gray Scale --------------------------------------------//
	clock_t start_cpu_GS, end_cpu_GS;
	float total_time_cpu_GS;
	start_cpu_GS = clock();

	unsigned char* grayImageData_cpu = grayscale_image_cpu(image.data, image.rows, image.cols, image.channels(), image.step);
	Mat processedImage_greyscale_cpu = Mat(image.rows, image.cols, CV_8UC1, grayImageData_cpu);
	//Window name and size
	namedWindow("Display CPU Gray Scale", WINDOW_AUTOSIZE);
	// Display grey scale Image
	imshow("Display CPU Gray Scale", processedImage_greyscale_cpu);

	end_cpu_GS = clock();
	//time count stops 
	total_time_cpu_GS = ((float)(end_cpu_GS - start_cpu_GS)) / CLOCKS_PER_SEC;
	//calulate total time
	printf("\nTime taken for converting image to greyscale in CPU: %f \n", total_time_cpu_GS);

	waitKey(0);
}

