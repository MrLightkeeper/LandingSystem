
#include <opencv.hpp>
#include <highgui.hpp>
#include <imgproc.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <vector>
#include <math.h>
#include <iostream>

using namespace cv;

int main(){


    bool mode = 1;
    CvCapture *capture1 = cvCaptureFromCAM(0);
       if( !capture1 ) return 1;
       cvNamedWindow("Video1");
       if (mode == 1){
       while(true)
       {
           //grab and retrieve each frames of the video sequentially
           IplImage* frame1 = cvQueryFrame( capture1 );

           cvShowImage( "Video1", frame1 );

           //wait for 40 milliseconds
           int c = cvWaitKey(40);

           //exit the loop if user press "Esc" key  (ASCII value of "Esc" is 27)
           if((char)c==27 ) break;
       }
}


return 0;
}
