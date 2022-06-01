#include <iostream>
#include <math.h>
#include <stdlib.h>
#include<string.h>
#include<mpi.h>
#include<tchar.h>
#include<Windows.h>
#include<omp.h>
#include<stdio.h>
#include <algorithm>
#include<msclr\marshal_cppstd.h>
#include <ctime>

#using <mscorlib.dll>
#using <System.dll>
#using <System.Drawing.dll>
#using <System.Windows.Forms.dll>

#define OPEN_MP 1
#define MPI 1

using namespace std;
using namespace msclr::interop;

int* inputImage(int* w, int* h, System::String^ imagePath) //put the size of image in w & h
{
	int* input;


	int OriginalImageWidth, OriginalImageHeight;

	//*******Read Image and save it to local arrayss***	
	//Read Image and save it to local arrayss

	System::Drawing::Bitmap BM(imagePath);

	OriginalImageWidth = BM.Width;
	OriginalImageHeight = BM.Height;
	*w = BM.Width;
	*h = BM.Height;
	int* Red = new int[BM.Height * BM.Width];
	int* Green = new int[BM.Height * BM.Width];
	int* Blue = new int[BM.Height * BM.Width];
	input = new int[BM.Height * BM.Width];
	for (int i = 0; i < BM.Height; i++)
	{
		for (int j = 0; j < BM.Width; j++)
		{
			System::Drawing::Color c = BM.GetPixel(j, i);

			Red[i * BM.Width + j] = c.R;
			Blue[i * BM.Width + j] = c.B;
			Green[i * BM.Width + j] = c.G;

			input[i * BM.Width + j] = ((c.R + c.B + c.G) / 3); //gray scale value equals the average of RGB values

		}

	}
	return input;
}

void createImage(int* image, int width, int height, int index)
{
	System::Drawing::Bitmap MyNewImage(width, height);


	for (int i = 0; i < MyNewImage.Height; i++)
	{
		for (int j = 0; j < MyNewImage.Width; j++)
		{

			//i * OriginalImageWidth + j
			if (image[i * width + j] < 0)
			{
				image[i * width + j] = 0;
			}
			if (image[i * width + j] > 255)
			{
				image[i * width + j] = 255;
			}
			System::Drawing::Color c = System::Drawing::Color::FromArgb(image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j], image[i * MyNewImage.Width + j]);
			MyNewImage.SetPixel(j, i, c);
		}
	}
	MyNewImage.Save("W:/HPC_ProjectTemplate/Data/Output/outputImage" + index + ".png");
	cout << "result Image Saved " << index << endl;
}

int ApplyMedianfilter(int* subImage, int size) {
	sort(subImage, subImage + size);

	if (size % 2 == 0)
		return (subImage[size / 2] + subImage[(size / 2) - 1]) / 2;
	else
		return subImage[size / 2];
}

vector<int> ImageProcessing(vector<int> image_after_padding, int kernalSize, int imageWidth, int imageHeight, int outputSize, int CHUNKSIZE) {
	int imageSize = imageHeight * imageWidth;
	int* subImage = new int[kernalSize * kernalSize];
	vector<int> medianValues(outputSize);
	int imageIndex = 0;

#if OPEN_MP
#pragma omp parallel for schedule (guided, CHUNKSIZE) firstprivate(imageIndex, subImage)
#endif // OPEN_MP
	for (int i = 0; i < imageSize; i++) {
		if ((i % imageWidth) + kernalSize > imageWidth) {
			continue;
		}
		//handle size kernal height
		if (i + (imageWidth * (kernalSize - 1)) < imageSize) {
			int index = i;
			int count = 0;

			for (int j = 0; j < (kernalSize * kernalSize); j++) {
				subImage[j] = image_after_padding[index];
				count++;
				index++;
				if (count == kernalSize) {
					index = index + (imageWidth - kernalSize);
					count = 0;
				}
			}
			medianValues[((imageIndex++) + (omp_get_thread_num() * CHUNKSIZE))] = ApplyMedianfilter(subImage, kernalSize * kernalSize);
		}
	}
	delete[] subImage;
	return medianValues;
}

vector <int> preprocessing(const int* image, int kernalSize, int* imageWidth, int* imageHeight, int numOfProcessors) {
	int origenalSize = (*imageWidth) * (*imageHeight);
	int numberOfRows = (*imageHeight) / numOfProcessors;
	int padSize = kernalSize / 2;
	vector <int> image_with_padding;
	vector<int> upperRow;
	vector<int> lowerRow;

	int rowNumber = 0;
	int index = 0;

	int imageCount = 0;
	for (int i = 1; i < numOfProcessors; i++) {
		for (int k = 0; k < padSize; k++) {
			for (int j = 0; j < (*imageWidth); j++) {

				rowNumber = (i * numberOfRows) + k;
				index = rowNumber * (*imageWidth);
				lowerRow.push_back(image[index + j]);

				rowNumber = (i * numberOfRows) - (padSize - k);
				index = rowNumber * (*imageWidth);
				upperRow.push_back(image[index + j]);
			}
		}
	}


	for (int m = 0; m < ((*imageWidth) + (kernalSize - 1)) * padSize; m++) {
		image_with_padding.push_back(0);
	}

	int rowsCounter = 0, procCounter = 0;

	for (int i = 0; i < (*imageHeight); i++) {
		for (int x = 0; x < padSize; x++)
			image_with_padding.push_back(0);
		for (int j = 0; j < (*imageWidth); j++) {
			image_with_padding.push_back(image[imageCount++]);
		}
		for (int x = 0; x < padSize; x++)
			image_with_padding.push_back(0);

		rowsCounter++;
		if (rowsCounter == numberOfRows && procCounter < numOfProcessors - 1) {
			for (int k = 0; k < padSize; k++) {
				for (int x = 0; x < padSize; x++)
					image_with_padding.push_back(0);

				for (int j = 0; j < (*imageWidth); j++) {
					image_with_padding.push_back(lowerRow[((procCounter + k) * (*imageWidth)) + j]);
				}

				for (int x = 0; x < padSize; x++)
					image_with_padding.push_back(0);
			}
			for (int k = 0; k < padSize; k++) {
				for (int x = 0; x < padSize; x++)
					image_with_padding.push_back(0);

				for (int j = 0; j < (*imageWidth); j++) {
					image_with_padding.push_back(upperRow[((procCounter + k) * (*imageWidth)) + j]);
				}

				for (int x = 0; x < padSize; x++)
					image_with_padding.push_back(0);
			}


			procCounter++;
			rowsCounter = 0;
		}
	}
	for (int m = 0; m < ((*imageWidth) + (kernalSize - 1)) * (padSize); m++) {
		image_with_padding.push_back(0);
	}
	(*imageWidth) += (kernalSize - 1);
	(*imageHeight) += (numOfProcessors * (kernalSize - 1));

	return image_with_padding;
}

int* VectorToArray(vector<int> vec) {
	int* arr = new int[vec.size()];

	for (int i = 0; i < vec.size(); i++) {
		arr[i] = vec[i];
	}

	return arr;
}

int main() {
	int start_s, stop_s, TotalTime = 0;
	int kernalSize = 3;

	int wRank = 0, wSize = 1;
#if MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &wSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
#endif

	int numOfImages = 1;
	int ImageWidth, ImageHeight;
	int PadImageWidth, padImageHeight;
	int* imageData = nullptr;
	int* sendDisplacement = new int[wSize] {0};
	int* recvDisplacement = new int[wSize] {0};
	int* sendCount = new int[wSize] {0};
	int* recvCount = new int[wSize] {0};

	int sendImageSubDatalength;
	int recvImageSubDatalength;
	vector<int>image_after_padding;
	int* image_after_padding_array = nullptr;

#if MPI
	if (wRank == 0)
	{
#endif
		System::String^ imagePath;
		std::string img;
		img = "W:/HPC_ProjectTemplate/Data/Input/5N_N_Salt_Pepper.PNG";

		cout << "Enter kernal size (Default = 3): ";
		cin >> kernalSize;
		cout << "Enter number of images (Default = 1): ";
		cin >> numOfImages;

		imagePath = marshal_as<System::String^>(img);
		imageData = inputImage(&ImageWidth, &ImageHeight, imagePath);
		PadImageWidth = ImageWidth;
		padImageHeight = ImageHeight;
		image_after_padding = preprocessing(imageData, kernalSize, &PadImageWidth, &padImageHeight, wSize);

		sendImageSubDatalength = (padImageHeight / wSize) * PadImageWidth;
		recvImageSubDatalength = (ImageHeight / wSize) * ImageWidth;

		int originalImageSize = (ImageWidth * ImageHeight);
		for (int n = 0; n < wSize; n++) {
			// Send Arrays
			sendDisplacement[n] = sendImageSubDatalength * n;
			if (n == (wSize - 1)) {
				sendCount[n] = (PadImageWidth * padImageHeight) - (sendImageSubDatalength * n);
			}
			else {
				sendCount[n] = sendImageSubDatalength;
			}


			// Recv Arrays
			recvDisplacement[n] = recvImageSubDatalength * n;
			if (n == (wSize - 1)) {
				recvCount[n] = originalImageSize - (recvImageSubDatalength * n);
			}
			else {
				recvCount[n] = recvImageSubDatalength;
			}
		}

		image_after_padding_array = VectorToArray(image_after_padding);
		start_s = clock();
#if MPI
	}
#endif
#if MPI
	MPI_Bcast(sendCount, wSize, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(recvCount, wSize, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ImageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ImageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&PadImageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&kernalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&numOfImages, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (wRank != 0)
		imageData = new int[ImageHeight * ImageWidth]{0};

	int* part_array = new int[sendCount[wRank]];
#else
	int* part_array = image_after_padding_array;
#endif
	for (int j = 0; j < numOfImages; j++)
	{

#if MPI
		MPI_Scatterv(image_after_padding_array, sendCount, sendDisplacement, MPI_INT, part_array, sendCount[wRank], MPI_INT, 0, MPI_COMM_WORLD);
#endif

		vector<int> result_image;

#if OPEN_MP
#pragma omp parallel
#endif

		result_image = ImageProcessing(vector<int>(part_array, part_array + sendCount[wRank]), kernalSize, PadImageWidth, sendCount[wRank] / PadImageWidth,
			recvCount[wRank], sendCount[wRank] / omp_get_num_threads());

#if MPI
		int* result_image_array = VectorToArray(result_image);
#else
		imageData = VectorToArray(result_image);
#endif

#if MPI
		MPI_Gatherv(result_image_array, recvCount[wRank], MPI_INT, imageData, recvCount, recvDisplacement, MPI_INT, 0, MPI_COMM_WORLD);
		if (wRank == 0)
		{
#endif

			createImage(imageData, ImageWidth, ImageHeight, j);

#if MPI

		}
		delete[] result_image_array;

		//MPI_Barrier(MPI_COMM_WORLD);
#endif
	}

	if (wRank == 0) {

		stop_s = clock();
		TotalTime += (stop_s - start_s) / CLOCKS_PER_SEC * 1000;
		cout << "time: " << TotalTime << endl;

		delete[] image_after_padding_array;
	}

	delete[] imageData;

#if MPI
	MPI_Finalize();
#endif

	return 0;

}