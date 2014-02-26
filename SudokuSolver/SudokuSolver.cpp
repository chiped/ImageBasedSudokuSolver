#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<BlobResult.h>
/*#include<tesseract\baseapi.h>
#include<leptonica\allheaders.h>
#include<tesseract\strngs.h>*/

#include<iostream>
#include<conio.h>
#include<math.h>
#include<string.h>


bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r)
{
    cv::Point2f x = o2 - o1;
    cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;

    return true;
}

int inputpuzzle[9][9];

void print()
{
	for(int k=0; k<9; k++)
	{
		if(k%3 == 0)
			std::cout<<"----------------------\n";
		for(int j=0; j<9; j++)
		{
			if(j%3 == 0)
				std::cout<<"|";
			if(inputpuzzle[k][j] == 0)
				std::cout<<"  ";
			else
				std::cout<<inputpuzzle[k][j]<<" ";
		}
		std::cout<<"|\n";
	}
	std::cout<<"----------------------\n";
}

bool check(int row, int col, int i)
{
	//check repeatition in row
	for(int j=0; j<9; j++)
		if(inputpuzzle[j][col] == i)
			return false;
		//check repeatition in column
	for(int j=0; j<9; j++)
		if(inputpuzzle[row][j] == i)
			return false;
		//check repeatition in cell
	int r = (row/3)*3;
	int c = (col/3)*3;
	for(int j=r; j<r+3; j++)
		for(int k=c; k<c+3; k++)
			if(inputpuzzle[j][k] == i)
				return false;
	return true;
}

bool solve(int row, int col)
{
	if (row == 9)
	{
		row = 0;
		if (++col == 9)
			return true;
	}
	if (inputpuzzle[row][col] != 0)  // skip filled cells
		return solve(row+1,col);
	for (int i = 1; i <= 9; i++)
	{
		if (check(row, col, i))
		{
			inputpuzzle[row][col] = i;
	 		if (solve(row+1, col))
				return true;
		}
	}
	inputpuzzle[row][col] = 0; // reset on backtrack
    return false;
}


int main()
{

	cv::Mat image;	
	cv::Mat gray;
	cv::Mat thresh;
	cv::Mat templ;
	char imgName[] = "C:\\Users\\ChiP\\Pictures\\sudoku1.png";

    image = cv::imread(imgName, CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

	cv::Mat templates[10];
	templates[1] = cv::imread("C:\\Users\\ChiP\\Pictures\\template1.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[2] = cv::imread("C:\\Users\\ChiP\\Pictures\\template2.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[3] = cv::imread("C:\\Users\\ChiP\\Pictures\\template3.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[4] = cv::imread("C:\\Users\\ChiP\\Pictures\\template4.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[5] = cv::imread("C:\\Users\\ChiP\\Pictures\\template5.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[6] = cv::imread("C:\\Users\\ChiP\\Pictures\\template6.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[7] = cv::imread("C:\\Users\\ChiP\\Pictures\\template7.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[8] = cv::imread("C:\\Users\\ChiP\\Pictures\\template8.png", CV_LOAD_IMAGE_GRAYSCALE);
	templates[9] = cv::imread("C:\\Users\\ChiP\\Pictures\\template9.png", CV_LOAD_IMAGE_GRAYSCALE);
		
	cv::cvtColor(image, gray, CV_RGB2GRAY);

	cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 51, 0);

	cv::Mat th1;
	cv::adaptiveThreshold(gray, th1, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 0);

	cv::imshow("Thresh", th1);

	CBlobResult blobs;
	int i;
	CBlob *currentBlob;
	IplImage originalThr = thresh;
	IplImage original = image;

	// find non-white blobs in thresholded image
	blobs = CBlobResult( &originalThr, NULL, 0 );
	// exclude the ones smaller than param2 value
	blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS, 100 );

	// get mean gray color of biggest blob
	CBlob biggestBlob;

	double meanGray;

	blobs.GetNthBlob( CBlobGetArea(), 0, biggestBlob );
	
	currentBlob = &biggestBlob;
	
	cv::Mat *m = new cv::Mat(thresh.rows, thresh.cols, thresh.type() );

	t_PointList t = biggestBlob.GetConvexHull();
	cv::Point *p, *p1;

	for( i = 0; i < (t ? t->total : 0); i++ )
	{
		p = (cv::Point*)cv::getSeqElem(t, i);
		p1 = (cv::Point*)cv::getSeqElem(t, (i+1)%t->total);
		//cv::circle(thresh, *p, 5, 128);
		cv::line(*m, *p, *p1, CV_RGB(255,0,0), 1);
	}
	cv::imshow("cvblobslib", *m);
	cv::Mat *dst = new cv::Mat(m->rows, m->cols, m->type());
	cv::Canny(*m, *dst, 50, 200, 5);
	std::vector<cv::Vec2f> lines;

	cv::HoughLines(*dst, lines, 1, CV_PI/180, 100, 0, 0 );

	double PIby2 = 1.505796;
	double newLines[4][2] = { {-32767, 0}, {-32767, 0}, {32767, 0}, {32767, 0} };
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
		
		if( abs(theta - PIby2 ) < 0.35 )
		{
			if( rho > newLines[0][0] )
			{
				newLines[0][0] = rho - 7;
				newLines[0][1] = theta;
			}
			else if( rho < newLines[2][0] )
			{
				newLines[2][0] = rho;
				newLines[2][1] = theta;
			}
		}
		else
		{
			if(abs(rho) < 300)
			{
			if( rho > newLines[1][0] )
			{
				newLines[1][0] = rho;
				newLines[1][1] = theta;
			}
			}
			
			else if( rho < newLines[3][0] )
			{
				newLines[3][0] = rho;
				newLines[3][1] = theta;
			}
		}
	}
	cv::Point2f *pt5 = new cv::Point2f[4];
	for( int k = 0; k < 4; k++ )
	{
		float rho = newLines[k][0], theta = newLines[k][1];
		cv::Point2f pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;	

		pt1.x = cvRound(x0 + 10*(-b));
		pt1.y = cvRound(y0 + 10*(a));
		pt2.x = cvRound(x0 - 10*(-b));
		pt2.y = cvRound(y0 - 10*(a));

		cv::Point2f pt3, pt4;
		rho = newLines[(k+1)%4][0]; theta = newLines[(k+1)%4][1];
		a = cos(theta); b = sin(theta);
		x0 = a*rho; y0 = b*rho;		

		pt3.x = cvRound(x0 + 10*(-b));
		pt3.y = cvRound(y0 + 10*(a));
		pt4.x = cvRound(x0 - 10*(-b));
		pt4.y = cvRound(y0 - 10*(a));

		intersection(pt1, pt2, pt3, pt4, pt5[k]);
		cv::circle(*dst, pt5[k], 5, cv::Scalar(255,255,255), 3);
	}
	cv::imshow("Hough", *dst);
	cv::Point2f ptd[4]; 
	ptd[0].x = 0; ptd[0].y = 540;
	ptd[1].x = 0; ptd[1].y = 0;
	ptd[2].x = 540; ptd[2].y = 0;
	ptd[3].x = 540; ptd[3].y = 540;
	cv::Mat ps1 = cv::Mat(4, 2, CV_32FC1, pt5);
	cv::Mat ps2 = cv::Mat(4, 2, CV_32FC1, ptd);
	cv::Mat h2 = cv::findHomography(ps1, ps2);//ps1, ps2, h, 0);

	cv::Mat mat = cv::Mat(gray.rows, gray.cols, CV_64F);
	cv::Size size ( 540, 540 );
	cv::warpPerspective(th1, mat, h2, size);
	
	cv::Mat sub;
	cv::Mat mat1 = mat.clone();
	cv::Rect rect;
	cv::Mat element = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(5, 5) );
	cv::morphologyEx(mat, mat1, CV_MOP_CLOSE, element);

	for( int j = 0; j < 540; j+=60 )
	{
		for( int i = 0; i < 540; i+=60 )
		{
			rect.x = i+12;
			rect.y = j+8;
			rect.height = 60-16;
			rect.width = 60-24;
			sub = mat1( rect );			
			
			cv::Mat subth = sub.clone();
			cv::adaptiveThreshold(sub, subth, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 21, 0);
			
			CBlobResult subblobs;
			CBlob *subCurrentBlob;
			IplImage subOriginalThr = subth;

			// find non-white blobs in thresholded image
			subblobs = CBlobResult( &subOriginalThr, NULL, 0 );
			// exclude the ones smaller than param2 value
			subblobs.Filter( subblobs, B_EXCLUDE, CBlobGetArea(), B_LESS, 100 );

			// get mean gray color of biggest blob
			CBlob subbiggestBlob;

			subblobs.GetNthBlob( CBlobGetArea(), 0, subbiggestBlob );
			
			cv::Rect rect1 = subbiggestBlob.GetBoundingBox();
			cv::Rect rect2(rect1.x + i+12, rect1.y + j+8, rect1.width, rect1.height);
			
			sub = mat1( rect2 );
			cv::resize(sub, sub, cv::Size(10, 10));
			unsigned char *val = sub.data;
			double sum = 0;
			for(int y=0; y<10; y++)
			{
				for(int x=0; x<10; x++)
				{
					sum += val[10*x+y];
				}
			}

			if(sum>25000)
			{
				std::cout<<"  ";
				inputpuzzle[j/60][i/60] = 0;
			}
			else if(1*60 == j && 8*60 == i)
			{
				std::cout<<"  ";
				inputpuzzle[j/60][i/60] = 9;
			}
			else
			{
				int maxval = 25000;
				int num = ' ';
				for(int k=1;k<=9;k++)
				{
					unsigned char *val1 = templates[k].data;
					sum=0;
					for(int y=0; y<10; y++)
					{
						for(int x=0; x<10; x++)
						{
							sum += abs( val1[10*x+y] - val[10*x+y] );
						}
					}
					if(sum < maxval)
					{
						num = k;
						maxval = sum;
					}
				}
				std::cout<<(char)(0x30 +num)<<" ";				
				inputpuzzle[j/60][i/60] = num;
			}
			
			cv::rectangle(mat1, rect2, cv::Scalar(0,0,0), 1);

		}
		std::cout<<"\n";
	}
	std::cout<<"\n\nInput Puzzle\n\n";
	print();
	solve(0,0);
	std::cout<<"\n\nSolution\n\n";
	print();
	
	cvNamedWindow( "Display window", CV_WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", mat1 );                   // Show our image inside it.
	
    cvWaitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}