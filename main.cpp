#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <vector>
#include <math.h>
#include <iostream>
#include <sys/time.h>


using namespace std;
using namespace cv;


IplImage* image = 0;
IplImage* src = 0;
IplImage* normalized = 0;
IplImage* gray = 0;
IplImage* binarized_1 = 0;


IplImage* centered = 0;
IplImage* filtered = 0;
IplImage* contours = 0;
IplImage* borders = 0;

IplImage* hsv = 0;
IplImage* dst = 0;
IplImage* h_range = 0;
IplImage* s_range = 0;
IplImage* v_range = 0;
IplImage* h_plane = 0;
IplImage* s_plane = 0;
IplImage* v_plane = 0;
IplImage* hsv_and = 0;

IplImage* r_range = 0;
IplImage* g_range = 0;
IplImage* b_range = 0;

IplImage* r_bin = 0;
IplImage* g_bin = 0;
IplImage* b_bin = 0;

unsigned char* data;
int element;
int coordX=0, coordY=0;
int xcc, ycc;
bool detected; //уНРЪ АШ НДХМ НАЕЙР МЮИДЕМ

int Hmin = 0;
int Hmax = 256;

int Smin = 0;
int Smax = 256;

int Vmin = 0;
int Vmax = 256;

int HSVmax = 256;
int col_arr[6]; //лЮЯЯХБ ОЮПЮЛЕРПНБ HSV

int sq_point_num = 0; //йНКХВЕЯРБН МЮИДЕММШУ НАЗЕЙРНБ
int cir_point_num = 0; //йНКХВЕЯРБН МЮИДЕММШУ НАЗЕЙРНБ

CvPoint centerPad; //йННПДХМЮРШ ЖЕМРПЮ ОКНЫЮДЙХ
CvPoint avSum; //яПЕДМЕЕ ГМЮВЕМХЕ ЙННПДХМЮР ЖЕМРПЮ ОКНЫЮДЙХ

CvPoint2D32f centerMass[1]; //жЕМРПЮ БЯЕУ МЮИДЕММШУ НАЗЕЙРНБ

int numOfPoints = cir_point_num;
    CvPoint pt1;
    CvPoint pt2;
    double rho;
    double eps;
    struct Obj
    {
        int dist;
        CvPoint point1;
        CvPoint point2;
        int pnum1;
        int pnum2;
    };
    struct Grp{

        Obj obj[100]; //нАЗЕЙРШ ЦПСООШ
        int numb; //мНЛЕП ЦПСООШ
        int size; //пЮГЛЕП ЦПСООШ
    };
    Grp group[200];
    Obj object[200];

int mode = 0; //пЕФХЛ ОНХЯЙЮ НАЗЕЙРНБ

#define LED_PAD		1;
#define WHITE_SQ	2;

//
// ТСМЙЖХХ-НАПЮАНРВХЙХ ОНКГСМЙНБ
//
void myTrackbarHmin(int pos) {
        Hmin = pos;
        cvInRangeS(h_plane, cvScalar(Hmin), cvScalar(Hmax), h_range);
}

void myTrackbarHmax(int pos) {
        Hmax = pos;
        cvInRangeS(h_plane, cvScalar(Hmin), cvScalar(Hmax), h_range);
}

void myTrackbarSmin(int pos) {
        Smin = pos;
        cvInRangeS(s_plane, cvScalar(Smin), cvScalar(Smax), s_range);
}

void myTrackbarSmax(int pos) {
        Smax = pos;
        cvInRangeS(s_plane, cvScalar(Smin), cvScalar(Smax), s_range);
}

void myTrackbarVmin(int pos) {
        Vmin = pos;
        cvInRangeS(v_plane, cvScalar(Vmin), cvScalar(Vmax), v_range);
}

void myTrackbarVmax(int pos) {
        Vmax = pos;
        cvInRangeS(v_plane, cvScalar(Vmin), cvScalar(Vmax), v_range);
}

double round(double number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}


void SquareByPoints(IplImage* _target)
{
    //мЮ БУНДЕ ЛЮЯЯХБ РНВЕЙ centerMass


    int i=0;
    int j=0;
    int objCount = 0;

    eps = 0.05;

    //нОПЕДЕКЕМХЕ ПЮЯЯРНЪМХИ Х ЯНГДЮМХЕ НАЗЕЙРНБ
    cout << '\n';
    for(i = 1; i <= numOfPoints; i++){
        for(j = i + 1; j <= numOfPoints; j++){
            objCount++;
            pt1 = cvPoint(centerMass[i].x, centerMass[i].y);
            pt2 = cvPoint(centerMass[j].x, centerMass[j].y);
            rho = sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y)); //ПЮЯЯРНЪМХЕ ЛЕФДС РНВЙЮЛХ
            object[objCount].dist = rho;
            object[objCount].point1 = pt1;
            object[objCount].point2 = pt2;
            object[objCount].pnum1 = i;
            object[objCount].pnum2 = j;
            cout << object[objCount].dist << ' ' << i << ' '<< j << '\n';
            //cout << pt1.x << ' ' << pt1.y << ' '<< pt2.x << ' '<< pt2.y << '\n';
        }
    }

    ////яНГДЮМХЕ ЛЮРПХЖШ ПЮЯЯРНЪМХИ
    //for(i = 1; i <= numOfPoints; i++){
    //	for(j = 1; j <= numOfPoints; j++){
    //		pt1 = cvPoint(centerMass[i].x, centerMass[i].y);
    //		pt2 = cvPoint(centerMass[j].x, centerMass[j].y);
    //		rho = sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y)); //ПЮЯЯРНЪМХЕ ЛЕФДС РНВЙЮЛХ
    //		M[i][j] = rho;
    //	}
    //}



    //яНПРХПНБЙЮ ОН ПЮЯЯРНЪМХЪЛ
    int mm;
    Obj tmp;
    for (i = objCount; i >= 0; i-- ){
        mm = 0;
        for (j = 0; j <= i; j++ ){
            if (object[j].dist > object[i].dist){
                mm = j;
                tmp = object[i];
                object[i] = object[mm];
                object[mm] = tmp;
            }
        }
    }

    //бШБНД ЯНПРХПНБЮММНЦН ЛЮЯЯХБЮ
    cout << "Sorted" << '\n';
    for (i = 1; i <= objCount; i++){
        cout << object[i].dist << '\n';
    }


    //пЮГАХБЙЮ НАЗЕЙРНБ МЮ ЦПСООШ ОН ХУ ДКХМЕ
    int countRebr[50];
    for (i = 0; i <= 9; i++){
        countRebr[i] = 1;
        //group[i].numb = 1;
        group[i].size = 1;
    }

    int k = 1; //мНЛЕП ЦПСООШ
    int maxk = 0; //йНКХВЕЯРБН ЦПСОО
    int l = 2; //яВЕРВХЙ НАЗЕЙРНБ Б ЦПСООЕ

    for (i = 2; i <= objCount; i++){
        group[k].obj[l-1] = object[i-1];
        if (abs(1 - object[i-1].dist / double(object[i].dist)) <= eps){
            countRebr[k]++;
            //group[k].numb++;

            group[k].obj[l] = object[i];
            group[k].size ++;
            l++;
            if (k >= maxk) maxk = k;
            //cvLine(_target, object[i-1].point1, object[i-1].point2, CV_RGB(20*k,50+30*k,0), 2, 8, 0);
            //cvLine(_target, object[i].point1, object[i].point2, CV_RGB(20*k,50+30*k,0), 2, 8, 0);
        }
        else{
            cout << "Num of vertices: " << countRebr[k] << '\n';
            k++;
            l = 2;
        }
    }

    for (k = 1; k <= maxk; k++){
        if (group[k].size >= 3){
            for (i = 2; i <= group[k].size; i++){
                cvLine(_target, object[i-1].point1, object[i-1].point2, CV_RGB(20*k,50+50*k,0), 2, 8, 0);
                cvLine(_target, object[i].point1, object[i].point2, CV_RGB(20*k,50+50*k,0), 2, 8, 0);
            }
        }
    }



    /*for (i = 1; i <= numOfPoints; i++){
        for(j = i + 1; j <= numOfPoints; j++){
            if (abs(object[i].dist - object[j].dist)/double(object[i].dist) <= eps){
                if (object[i].pnum1 == object[j].pnum1){
                    for (int k = 1; k <= numOfPoints; k++){
                        cout << "For reached \n";
                        if ((object[k].pnum1 == object[i].pnum2 && object[k].pnum2 == object[j].pnum2) || (object[k].pnum1 == object[j].pnum2 && object[k].pnum2 == object[i].pnum2)){
                            cout << "triangle" << '\n';
                            cvLine(_target, object[i].point1, object[i].point2, CV_RGB(0,150,0), 2, 8, 0);
                            cvLine(_target, object[j].point1, object[j].point2, CV_RGB(0,150,0), 2, 8, 0);
                            cvLine(_target, object[k].point1, object[k].point2, CV_RGB(0,150,0), 2, 8, 0);
                        }
                    }
                }
            }
        }
    }*/






    //for(i = 2; i <= objCount; i++){
    //	j=i-1;
    //		if (abs(object[i].dist - object[j].dist)/double(object[i].dist) <= eps){
    //

    //			cx1 = (object[i].point1.x + object[i].point2.x)/2;
    //			cy1 = (object[i].point1.y + object[i].point2.y)/2;

    //			cx2 = (object[j].point1.x + object[j].point2.x)/2;
    //			cy2 = (object[j].point1.y + object[j].point2.y)/2;

    //			cout << ' ' << j << ' '<< i << ' ' << double((object[i].dist - object[j].dist)/double(object[i].dist)) << ' '<< cx1 << ' '<< cy1 << ' '<< cx2 << ' '<< cy2 <<  '\n';

    //			if (abs(cx1 - cx2)/double(cx1) <= eps && abs(cy1 - cy2)/double(cy1) <= eps){
    //				cout << "rectangle" << ' ' << abs((cx1 - cx2)/double(cx1)) << ' ' << abs(cy1 - cy2)/double(cy1) << ' ' << j << ' '<< i << '\n';
    //				//cvLine(_target, object[i].point1, object[i].point2, CV_RGB(0,150,0), 2, 8, 0);
    //				//cvLine(_target, object[j].point1, object[j].point2, CV_RGB(0,150,0), 2, 8, 0);
    //			}
    //		}
    //}



}

void findSquare(IplImage* _image,IplImage* _target , IplImage* _color)
{
        assert(_image!=0);

        IplImage* bin = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1);

        // ЙНМБЕПРХПСЕЛ Б ЦПЮДЮЖХХ ЯЕПНЦН
        cvConvertImage(_image, bin, CV_BGR2GRAY);
        // МЮУНДХЛ ЦПЮМХЖШ
        cvCanny(bin, bin, 60, 255, 3); //60-255
        cvNamedWindow( "Canny", 1 );
        cvShowImage("Canny", bin);

        // УПЮМХКХЫЕ ОЮЛЪРХ ДКЪ ЙНМРСПНБ
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* contours=0;

        // МЮУНДХЛ ЙНМРСПШ
        //int contoursCont = cvFindContours( bin, storage,&contours,sizeof(CvContour),CV_RETR_LIST,CV_LINK_RUNS,cvPoint(0,0));


        //CvMemStorage* storage_l = cvCreateMemStorage(0);
  //      CvSeq* lines = 0;
  //      int i = 0;
        //IplImage* color_dst = cvCreateImage( cvGetSize(src), 8, 3 );
        //lines = cvHoughLines2( bin, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 50, 50, 10 );
        // МЮПХЯСЕЛ МЮИДЕММШЕ КХМХХ
  //      for( i = 0; i < lines->total; i++ ){
  //              CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
  //              cvLine( centered, line[0], line[1], CV_RGB(255,0,0), 3, CV_AA, 0 );
  //      }
        //cvShowImage("With Center", centered);

        // ДКЪ НРЛЕРЙХ ЙНМРСПНБ
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 2.0);
        char buf[128];
        int counter=0;

        //assert(contours!=0);

        // НАУНДХЛ БЯЕ ЙНМРСПШ
        for( CvSeq* current = contours; current != NULL; current = current->h_next ){
                // БШВХЯКЪЕЛ ОКНЫЮДЭ Х ОЕПХЛЕРП ЙНМРСПЮ
                double area = fabs(cvContourArea(current));
                double perim = cvContourPerimeter(current);


                // 1/4*CV_PI = 0,079577
                // square = 0,0625

                if ( area / (perim * perim) > 0.0555 && area / (perim * perim)< 0.0695 && area >= 10*10){ // Б 10% ХМРЕПБЮКЕ

                        CvPoint2D32f center;
                        float radius=0;

                        // МЮУНДХЛ ОЮПЮЛЕРПШ НЙПСФМНЯРХ
                        cvMinEnclosingCircle(current, & center, &radius);

                        //нРПХЯНБЙЮ ОПЪЛНСЦНКЭМШУ ПЮЛНЙ
                        CvBox2D rect = cvMinAreaRect2(current);
                        CvPoint2D32f rect_points[4];
                        cvBoxPoints(rect, rect_points);
                        int boxArea = rect.size.height * rect.size.width;
                        for( int j = 0; j < 4; j++ ){
                            cvLine( _target, cvPoint(rect_points[j].x,rect_points[j].y), cvPoint(rect_points[(j+1)%4].x,rect_points[(j+1)%4].y), CV_RGB(255,255,255), 1, 8 );
                            cvLine( _color, cvPoint(rect_points[j].x,rect_points[j].y), cvPoint(rect_points[(j+1)%4].x,rect_points[(j+1)%4].y), CV_RGB(255,255,255), 1, 8 );
                        }

                        // МЮПХЯСЕЛ ЙНМРСП
                        if ( abs(area - boxArea)/area <= 0.1 && abs(rect.size.height - rect.size.width)/rect.size.height <= 0.15){
                        //if (area >= 10*10  && area > (2*(radius * radius)* 0.95) && area < (2*(radius * radius)* 1.1)){

                                cvDrawContours(_target, current, cvScalar(255, 255, 255), cvScalar(0, 255, 0), 0, -1, 8);

                                // БШБНДХЛ ЕЦН МНЛЕП
                                //CvPoint2D32f point; float rad;
                                //cvMinEnclosingCircle(current,&point,&rad); // ОНКСВХЛ НЙПСФМНЯРЭ ЯНДЕПФЮЫСЧ ЙНМРСП
                                sprintf(buf, "%i", ++counter);
                                cvPutText(_target, buf, cvPointFrom32f(center), &font, CV_RGB(255,255,255));
                                centerMass[counter] = center;
                        }
                }
        }
        sq_point_num = counter;

        // НЯБНАНФДЮЕЛ ПЕЯСПЯШ
        cvReleaseMemStorage(&storage);
        cvReleaseImageHeader(&bin);
}

void findCircles(IplImage* _image,IplImage* _target)
{
        assert(_image!=0);

        IplImage* bin = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1);

        // ЙНМБЕПРХПСЕЛ Б ЦПЮДЮЖХХ ЯЕПНЦН
        cvConvertImage(_image, bin, CV_BGR2GRAY);
        // МЮУНДХЛ ЦПЮМХЖШ
        //cvCanny(bin, bin, 50, 250);
        /*cvNamedWindow( "Canny", 1 );
        cvShowImage("Canny", bin);*/

        // УПЮМХКХЫЕ ОЮЛЪРХ ДКЪ ЙНМРСПНБ
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvSeq* contours=0;

        // МЮУНДХЛ ЙНМРСПШ
        cvFindContours( bin, storage,&contours,sizeof(CvContour),CV_RETR_LIST,CV_LINK_RUNS,cvPoint(0,0));

        // ДКЪ НРЛЕРЙХ ЙНМРСПНБ
        CvFont font;
        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 2.0);
        char buf[128];
        int counter=0;

        //assert(contours!=0);

        // НАУНДХЛ БЯЕ ЙНМРСПШ
        for( CvSeq* current = contours; current != NULL; current = current->h_next )
        {
            // БШВХЯКЪЕЛ ОКНЫЮДЭ Х ОЕПХЛЕРП ЙНМРСПЮ
            double area = fabs(cvContourArea(current));
            double perim = cvContourPerimeter(current);

            // 1/4*CV_PI = 0,079577
            if ( area / (perim * perim) > 0.059 && area / (perim * perim)< 0.087 ){

                // Drawing circles
                float radius;
                CvPoint2D32f center;
                cvMinEnclosingCircle(current, & center, &radius);
                double circArea = 2 * 3.14 * radius;
                cvCircle(_target, cvPoint(center.x,center.y), radius, CV_RGB(255,255,255), 1, 8);

                // МЮПХЯСЕЛ ЙНМРСП
                if ((area >= 2*2) &&  abs(1 - perim / circArea) <= 0.2)
                {
                    cvDrawContours(_target, current, cvScalar(255, 255, 255), cvScalar(0, 255, 0), 0, -1, 8);
                    // БШБНДХЛ ЕЦН МНЛЕП
                    //CvPoint2D32f point; float rad;
                    //cvMinEnclosingCircle(current,&point,&rad); // ОНКСВХЛ НЙПСФМНЯРЭ ЯНДЕПФЮЫСЧ ЙНМРСП
                    sprintf(buf, "%i", ++counter);
                    cvPutText(_target, buf, cvPointFrom32f(center), &font, CV_RGB(255,255,255));

                    centerMass[counter] = center;
                }
            }
        }
        cir_point_num = counter;

        // НЯБНАНФДЮЕЛ ПЕЯСПЯШ
        cvReleaseMemStorage(&storage);
        cvReleaseImage(&bin);
}

void findCenter(IplImage* _image, IplImage* _bin, int point_num)
{
        centered = cvCreateImage(cvSize(_image->width, _image->height), _image -> depth, 3);
        cvCvtColor(_bin, centered, CV_GRAY2RGB);

        data = (unsigned char*)(_bin->imageData);

        CvMat *mat = cvCreateMat(_image->height,_image->width,CV_8UC1 );
        cvSetData(mat,data,_bin->width);
        int J=0, I=0, count=0;
        int xc,yc;

        int bufimax=0,bufjmax=0,bufimin=(centered->width)+1,bufjmin=(centered->height)+1;

        //НРПХЯНБЙЮ ЖЕМРПНБ ЙНМРСПНБ
        for( int i = 1; i <= point_num; i++)
        {
                cvCircle(centered, cvPoint(centerMass[i].x,centerMass[i].y), 2, CV_RGB(250,0,0),1, 8);

        }
        //printf("Size %d \n", point_num);
        CvPoint* pts = new CvPoint[sizeof(CvPoint)*point_num];
        for(int i = 1; i <= point_num; i++)
        {
                pts[i-1]= cvPoint(centerMass[i].x, centerMass[i].y);
                //printf("%d, %d \n", pts[i-1].x, pts[i-1].y);
        }
        cvPolyLine(centered, &pts, &point_num,1,true,CV_RGB(250,0,0),1,8,0);

        for( int k = 0; k < point_num; k++)
        {
                    J=J+pts[k].y;
                    I=I+pts[k].x;
                    count++;
                    if(sqrt(pts[k].x*pts[k].x+pts[k].y*pts[k].y)<(sqrt(bufimin*bufimin+bufjmin*bufjmin)))
                    {
                            bufimin=pts[k].x;
                            bufjmin=pts[k].y;
                    }
                    if(sqrt(pts[k].x*pts[k].x+pts[k].y*pts[k].y)>(sqrt(bufimax*bufimax+bufjmax*bufjmax)))
                    {
                            bufimax=pts[k].x;
                            bufjmax=pts[k].y;
                    }
        }


        if(count!=0)
        {
                xc=I/point_num;
                yc=J/point_num;

                //printf("Center %d, %d - Num = %d \n", xc, yc, point_num);

                cvCircle(centered,cvPoint(xc, yc),5,CV_RGB(250,0,0),1,8);
                centerPad = cvPoint(xc, yc);
                //cvShowImage("With Center",centered);
                detected = true;
        }
        else
        {
                CvFont font;
                cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 3.0, 3.0,0,3);
                cvPutText(centered,  "No target", cvPoint(centered->width /3, centered->height /2), &font, CV_RGB(255,0,0));
                centerPad = cvPoint(0, 0);
                //cvShowImage("With Center",centered);
                detected = false;
        }
}

IplImage* createBin(IplImage* _image, int _low, int _high) //аХМЮПХГЮЖХЪ ХГНАПЮФЕМХЪ
{
    IplImage* _bin = 0;
    _bin = cvCreateImage(cvSize(_image->width, _image->height), _image -> depth, 1);
    cvThreshold(_image, _bin, _low, _high, CV_THRESH_BINARY);
    //cvAdaptiveThreshold(gray, binarized_1, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 3,0.0);
    return _bin;
}

void hsvConvert(IplImage* _image, IplImage* dst, bool _manual, char _color){ //Convert image to HSV format

        // НОПЕДЕКЕМХЕ ОЮПЮЛЕРПНБ ЖБЕРЮ

        if(_color == 'b')
        {
                col_arr[0] = 90; //h_min
                col_arr[1] = 120; //h_max
                col_arr[2] = 70; //s_min
                col_arr[3] = 255; //s_max
                col_arr[4] = 190; //v_min
                col_arr[5] = 255; //v_max
        }
        else if(_color == 'r')
        {
                col_arr[0] = 150; //h_min
                col_arr[1] = 180; //h_max
                col_arr[2] = 40; //s_min
                col_arr[3] = 255; //s_max
                col_arr[4] = 150; //v_min
                col_arr[5] = 255; //v_max
        }
        else if(_color == 'd')
        {
                col_arr[0] = 80; //h_min
                col_arr[1] = 105; //h_max
                col_arr[2] = 0; //s_min
                col_arr[3] = 160; //s_max
                col_arr[4] = 180; //v_min
                col_arr[5] = 255; //v_max
        }
        else if(_color == 'o')
        {
                col_arr[0] = 150; //h_min
                col_arr[1] = 160; //h_max
                col_arr[2] = 120; //s_min
                col_arr[3] = 255; //s_max
                col_arr[4] = 170; //v_min
                col_arr[5] = 220; //v_max
        }
        else
        {
                col_arr[0] = 0; //h_min
                col_arr[1] = 255; //h_max
                col_arr[2] = 0; //s_min
                col_arr[3] = 255; //s_max
                col_arr[4] = 0; //v_min
                col_arr[5] = 255; //v_max
        }

        // ЯНГДЮ╦Л ЙЮПРХМЙХ
        //if (dst!=0) cvReleaseImage(&dst);

        hsv = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 3 );
        h_plane = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );
        s_plane = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );
        v_plane = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );
        h_range = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );
        s_range = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );
        v_range = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );
       // IplImage* tmp = cvCreateImage( cvGetSize(_image), IPL_DEPTH_8U, 1 );


        //  ЙНМБЕПРХПСЕЛ Б HSV
        cvCvtColor( _image, hsv, CV_BGR2HSV );
        // ПЮГАХБЮЕЛ МЮ НРЕКЭМШЕ ЙЮМЮКШ
        cvSplit( hsv, h_plane, s_plane, v_plane, 0 );

        //char v = cvWaitKey(33);


        if((_manual == true))
        {

                // НЙМЮ ДКЪ НРНАПЮФЕМХЪ ЙЮПРХМЙХ
                cvNamedWindow("original",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("H",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("S",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("V",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("H range",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("S range",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("V range",CV_WINDOW_AUTOSIZE);
                cvNamedWindow("hsv and",CV_WINDOW_AUTOSIZE);

                //
                // НОПЕДЕКЪЕЛ ЛХМХЛЮКЭМНЕ Х ЛЮЙЯХЛЮКЭМНЕ ГМЮВЕМХЕ
                // С ЙЮМЮКНБ HSV
                double framemin=0;
                double framemax=0;

                cvMinMaxLoc(h_plane, &framemin, &framemax);
                //printf("[H] %f x %f\n", framemin, framemax );
                Hmin = framemin;
                Hmax = framemax;
                cvMinMaxLoc(s_plane, &framemin, &framemax);
                //printf("[S] %f x %f\n", framemin, framemax );
                Smin = framemin;
                Smax = framemax;
                cvMinMaxLoc(v_plane, &framemin, &framemax);
                //printf("[V] %f x %f\n", framemin, framemax );
                Vmin = framemin;
                Vmax = framemax;

                cvCreateTrackbar("Hmin", "H range", &Hmin, HSVmax, myTrackbarHmin);
                cvCreateTrackbar("Hmax", "H range", &Hmax, HSVmax, myTrackbarHmax);
                cvCreateTrackbar("Smin", "S range", &Smin, HSVmax, myTrackbarSmin);
                cvCreateTrackbar("Smax", "S range", &Smax, HSVmax, myTrackbarSmax);
                cvCreateTrackbar("Vmin", "V range", &Vmin, HSVmax, myTrackbarVmin);
                cvCreateTrackbar("Vmax", "V range", &Vmax, HSVmax, myTrackbarVmax);

                while(true){

                        // ОНЙЮГШБЮЕЛ ЙЮПРХМЙС
                        cvShowImage("original",_image);

                        cvShowImage( "H", h_plane );
                        cvShowImage( "S", s_plane );
                        cvShowImage( "V", v_plane );

                        cvShowImage( "H range", h_range );
                        cvShowImage( "S range", s_range );
                        cvShowImage( "V range", v_range );

                        // ЯЙКЮДШБЮЕЛ
                        cvAnd(h_range, s_range, dst);
                        cvAnd(dst, v_range, dst);

                        cvShowImage( "hsv and", dst);

                        char c = cvWaitKey(33);
                        if (c == 27) { // ЕЯКХ МЮФЮРЮ ESC - БШУНДХЛ
                                break;
                        }
                }
        }
        else
        {
                Hmin = col_arr[0];
                Hmax = col_arr[1];
                cvInRangeS(h_plane, cvScalar(Hmin), cvScalar(Hmax), h_range);
                Smin = col_arr[2];
                Smax = col_arr[3];
                cvInRangeS(s_plane, cvScalar(Smin), cvScalar(Smax), s_range);
                Vmin = col_arr[4];
                Vmax = col_arr[5];
                cvInRangeS(v_plane, cvScalar(Vmin), cvScalar(Vmax), v_range);

                // ЯЙКЮДШБЮЕЛ
                cvAnd(h_range, s_range, dst);
                cvAnd(dst, v_range, dst);

        }
        cvReleaseImage(&hsv);
        cvReleaseImage(&h_plane);
        cvReleaseImage(&s_plane);
        cvReleaseImage(&v_plane);
        cvReleaseImage(&h_range);
        cvReleaseImage(&s_range);
        cvReleaseImage(&v_range);
}


int main()
{
        CvCapture *capture = cvCreateCameraCapture(0);
        if( !capture ){
            std::cout << "Unable to read stream from specified device." << std::endl;
            return 1;
        }
        else std::cout << "[i] Capture opened" << std::endl;


        cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 800);
        cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 600);


        double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
        double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
        std::cout << "[i] Resolution " << width << " x " << height << std::endl;
        std::cout << "[i] press Esc for quit!" << std::endl;


        //For searching average position of pad
        CvPoint avCenter;
        int avCount = 0;
        int aoiSize = 100;

        //Preferences

        bool flManHSV = false;  //Manual set up HSV parameters
        bool flManMode = 0;     //Manual or automative choosing mode
        mode = 1;               //Choose mode

        timeval startTime;
        gettimeofday(&startTime, NULL);

        start:                  //This is for goto()


        if (mode == 1)          //Searching by LEDs
        {
                while(true)
                {

                        IplImage *image = cvQueryFrame(capture);
                        assert( image !=0 );
                        normalized = cvCloneImage( image);

                        timeval beginTime;
                        gettimeofday(&beginTime, NULL);



                        //ТХКЭРПЮЖХЪ
                        cvSmooth(normalized, normalized, CV_GAUSSIAN, 5, 5);

                        cvNamedWindow("normalized", CV_WINDOW_AUTOSIZE);
                        cvShowImage("normalized", normalized);
                        cvMoveWindow("normalized",10,10);

                        //ЙПЮЯМШИ Х ЯХМХИ ЯБЕРНДХНДШ

                        binarized_1 = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 1);
                        //cvAdd(hsvConvert(normalized, false, 'r'), hsvConvert(normalized, false, 'b'), binarized_1);
                        hsvConvert(normalized, binarized_1, flManHSV, 'd');

                        cvNamedWindow("Bin");
                        cvShowImage("Bin", binarized_1);


                        //ЙНМРСПШ
                        filtered = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 3);
                        cvCvtColor(binarized_1, filtered, CV_GRAY2RGB);


                        contours = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 1);
                        cvZero(contours);
                        findCircles(filtered, contours);

                        cvNamedWindow("Contours");
                        cvShowImage("Contours", contours);


                        //ОНХЯЙ ЖЕМРПЮ

                        findCenter(normalized, contours, cir_point_num);

                        //ЛНДСКЭ НРПХЯНБЙХ ХМТНПЛЮЖХХ

                        timeval endTime;
                        gettimeofday(&endTime, NULL);
                        int frameSecTime = endTime.tv_sec - beginTime.tv_sec;
                        int frameMilliSecTime = endTime.tv_usec / 1000 - beginTime.tv_usec / 1000;
                        double elapsed_msecs = (frameSecTime * 1000 + frameMilliSecTime);

                        CvFont font;
                        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 2.0);
                        char buf[128];

                        sprintf(buf, "FPS %.2f", 1000 / elapsed_msecs);
                        cvPutText(centered,  buf, cvPoint(5, centered->height - 10), &font, CV_RGB(255,0,0));

                        sprintf(buf, "X = %4d", -(centerPad.y - (centered->height)/2));
                        cvPutText(centered,  buf, cvPoint(centered -> width - 130, centered->height - 30), &font, CV_RGB(0,255,0));
                        sprintf(buf, "Y = %4d", centerPad.x-(centered->width)/2);
                        cvPutText(centered,  buf, cvPoint(centered -> width - 130, centered->height - 10), &font, CV_RGB(0,255,0));


                        timeval currentTime;
                        gettimeofday(&currentTime, NULL);
                        int currentSecTime = currentTime.tv_sec - startTime.tv_sec;
                        int currentMilliSecTime = currentTime.tv_usec / 1000 - startTime.tv_usec / 1000;
                        double time = (currentSecTime * 1000) + currentMilliSecTime;

                        sprintf(buf, "Time %.2f sec", time / 1000);

                        cvPutText(centered,  buf, cvPoint(5, centered->height - 30), &font, CV_RGB(0,255,0));
                        cvPutText(centered,  "Looking for LED", cvPoint(5, 25), &font, CV_RGB(0,255,0));

                        cvLine(centered, cvPoint(centered->width/2 - 10,centered->height/2),cvPoint(centered->width/2 + 10,centered->height/2), CV_RGB(0,255,0), 2, 8, 0);
                        cvLine(centered, cvPoint(centered->width/2,centered->height/2 - 10),cvPoint(centered->width/2,centered->height/2 + 10), CV_RGB(0,255,0), 2, 8, 0);

                        //ОНХЯЙ ЯПЕДМЕЦН ГМЮВЕМХЪ ЖЕМРПЮ ОКНЫЮДЙХ
                        if(detected)
                        {
                            avCount++;
                            avSum.x += centerPad.x;
                            avSum.y += centerPad.y;
                            avCenter.x = avSum.x / avCount;
                            avCenter.y = avSum.y / avCount;
                            if (avCount >= 50)
                            {
                                avSum.x = 0;
                                avSum.y = 0;
                                avCount = 0;
                            }
                            cvCircle(centered,avCenter,5,CV_RGB(0,0,250),1,8);
                            cvRectangle(centered, cvPoint(avCenter.x - aoiSize/2,avCenter.y - aoiSize/2), cvPoint(avCenter.x + aoiSize/2,avCenter.y + aoiSize/2),CV_RGB(0,0,250),1,8);
                        }

                        //SquareByPoints(centered);

                        cvShowImage("With Center",centered);


                        cvReleaseImage(& normalized);
                        cvReleaseImage(& binarized_1);
                        cvReleaseImage(& centered);
                        cvReleaseImage(& filtered);
                        cvReleaseImage(& contours);
                        cvReleaseImageHeader(& image);


                        char c = cvWaitKey(33);
                        if (c == 27) { // МЮФЮРЮ ESC
                                break;
                        }
                        if (c == 115){ // МЮФЮРЮ s
                            SquareByPoints(centered);
                        }
                        if (c == 109) { // МЮФЮРЮ m
                                flManHSV = true;
                        }
                        else flManHSV = false;

                        if ((cir_point_num < 3 && flManMode == false) || c == 120){ // ЙКЮБХЬЮ У
                                mode = WHITE_SQ;
                                goto start;
                        }
                }

                // НЯБНАНФДЮЕЛ ПЕЯСПЯШ
                cvReleaseCapture( &capture );
                // СДЮКЪЕЛ НЙМН
                cvDestroyWindow("With Center");
                cvDestroyWindow("normalized");
                cvDestroyWindow("Bin");
        }
        else if (mode == 2) //ОНХЯЙ ОН ЙБЮДПЮРС

        {
                while(true)
                {
                        IplImage *image = cvQueryFrame(capture);
                        assert( image !=0 );

                        normalized = cvCloneImage( image);
                        timeval beginTime;
                        gettimeofday(&beginTime, NULL);

                        //ТХКЭРПЮЖХЪ
                        cvSmooth(normalized, normalized, CV_GAUSSIAN, 3, 3);

                        cvNamedWindow("normalized", CV_WINDOW_AUTOSIZE);
                        cvShowImage("normalized", normalized);
                        cvMoveWindow("normalized",10,10);

                        binarized_1 = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 1);
                        cvCvtColor(normalized, binarized_1, CV_RGB2GRAY);
                        //binarized_1 = hsvConvert(normalized, true, 'r');


                        cvNamedWindow( "Bin", 1 );
                        cvShowImage("Bin", binarized_1);


                        //ЙНМРСПШ
                        filtered = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 3);
                        cvCvtColor(binarized_1, filtered, CV_GRAY2RGB);


                        contours = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 1);
                        cvZero(contours);
                        findSquare(filtered, contours, normalized);

                        //ОНХЯЙ ЖЕМРПЮ

                        findCenter(normalized, contours, sq_point_num);


                        //ЛНДСКЭ НРПХЯНБЙХ ХМТНПЛЮЖХХ

                        timeval endTime;
                        gettimeofday(&endTime, NULL);
                        int frameSecTime = endTime.tv_sec - beginTime.tv_sec;
                        int frameMilliSecTime = endTime.tv_usec / 1000 - beginTime.tv_usec / 1000;
                        double elapsed_msecs = (frameSecTime * 1000 + frameMilliSecTime);

                        CvFont font;
                        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 2.0);
                        char buf[128];
                        sprintf(buf, "FPS %.2f", 1000/elapsed_msecs);
                        cvPutText(centered,  buf, cvPoint(5, centered->height - 10), &font, CV_RGB(255,0,0));

                        sprintf(buf, "X = %4d", -(centerPad.y - (centered->height)/2));
                        cvPutText(centered,  buf, cvPoint(centered -> width - 130, centered->height - 30), &font, CV_RGB(0,255,0));
                        sprintf(buf, "Y = %4d", centerPad.x-(centered->width)/2);
                        cvPutText(centered,  buf, cvPoint(centered -> width - 130, centered->height - 10), &font, CV_RGB(0,255,0));

                        timeval currentTime;
                        gettimeofday(&currentTime, NULL);
                        int currentSecTime = currentTime.tv_sec - startTime.tv_sec;
                        int currentMilliSecTime = currentTime.tv_usec / 1000 - startTime.tv_usec / 1000;
                        double time = (currentSecTime * 1000) + currentMilliSecTime;

                        sprintf(buf, "Time %.2f sec", double(time)/1000);
                        cvPutText(centered,  buf, cvPoint(5, centered->height - 30), &font, CV_RGB(0,255,0));
                        cvPutText(centered,  "Looking for Square", cvPoint(5, 25), &font, CV_RGB(0,255,0));

                        cvLine(centered, cvPoint(centered->width/2 - 10,centered->height/2),cvPoint(centered->width/2 + 10,centered->height/2), CV_RGB(0,255,0), 2, 8, 0);
                        cvLine(centered, cvPoint(centered->width/2,centered->height/2 - 10),cvPoint(centered->width/2,centered->height/2 + 10), CV_RGB(0,255,0), 2, 8, 0);

                        //ОНХЯЙ ЯПЕДМЕЦН ГМЮВЕМХЪ ЖЕМРПЮ ОКНЫЮДЙХ
                        if(detected)
                        {
                            avCount++;
                            avSum.x += centerPad.x;
                            avSum.y += centerPad.y;
                            avCenter.x = avSum.x / avCount;
                            avCenter.y = avSum.y / avCount;
                            if (avCount >= 50)
                            {
                                avSum.x = 0;
                                avSum.y = 0;
                                avCount = 0;
                            }
                            cvCircle(centered,avCenter,5,CV_RGB(0,0,250),1,8);
                            cvRectangle(centered, cvPoint(avCenter.x - aoiSize/2,avCenter.y - aoiSize/2), cvPoint(avCenter.x + aoiSize/2,avCenter.y + aoiSize/2),CV_RGB(0,0,250),1,8);
                        }


                        cvShowImage("With Center",centered);

                        cvReleaseImage(& normalized);
                        cvReleaseImage(& binarized_1);
                        cvReleaseImage(& centered);
                        cvReleaseImage(& filtered);
                        cvReleaseImage(& contours);
                        cvReleaseImageHeader(& image);


                        char c = cvWaitKey(33);
                        if (c == 27) { // МЮФЮРЮ ESC
                                break;
                        }
                        if ((sq_point_num == 0 && flManMode == false) || (c == 120)){ // ЙКЮБХЬЮ У
                                mode = LED_PAD;
                                goto start;
                        }
                }
                // НЯБНАНФДЮЕЛ ПЕЯСПЯШ
                cvReleaseCapture( &capture );

                // СДЮКЪЕЛ НЙМН
                cvDestroyWindow("resized");
                cvDestroyWindow("normalized");

        }
        if (mode == 3) //ОНХЯЙ НПЮМФЕБНЦН ЙБЮДПЮРЮ
        {
                while(true)
                {
                        IplImage *image = cvQueryFrame(capture);
                        assert( image !=0 );
                        normalized = cvCloneImage( image);
                        timeval beginTime;
                        gettimeofday(&beginTime, NULL);

                        //ТХКЭРПЮЖХЪ
                        cvSmooth(normalized, normalized, CV_GAUSSIAN, 5, 5);

                        cvNamedWindow("normalized", CV_WINDOW_AUTOSIZE);

                        cvMoveWindow("normalized",10,10);

                        //НПЮМФЕБШИ ЙБЮДПЮР

                        binarized_1 = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 1);
                        //cvAdd(hsvConvert(normalized, false, 'r'), hsvConvert(normalized, false, 'b'), binarized_1);
                        hsvConvert(normalized, binarized_1, flManHSV, 'o');

                        cvNamedWindow( "Bin", 1 );
                        cvShowImage("Bin", binarized_1);


                        //ЙНМРСПШ
                        filtered = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 3);
                        cvCvtColor(binarized_1, filtered, CV_GRAY2RGB);


                        contours = cvCreateImage(cvSize(normalized->width, normalized->height), normalized -> depth, 1);
                        cvZero(contours);
                        findSquare(filtered, contours, normalized);

                        //ОНХЯЙ ЖЕМРПЮ

                        findCenter(normalized, contours, cir_point_num);

                        //ЛНДСКЭ НРПХЯНБЙХ ХМТНПЛЮЖХХ

                        timeval endTime;
                        gettimeofday(&endTime, NULL);
                        int frameSecTime = endTime.tv_sec - beginTime.tv_sec;
                        int frameMilliSecTime = endTime.tv_usec / 1000 - beginTime.tv_usec / 1000;
                        double elapsed_msecs = (frameSecTime * 1000 + frameMilliSecTime);

                        CvFont font;
                        cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 2.0);
                        char buf[128];

                        sprintf(buf, "FPS %.2f", 1000/elapsed_msecs);
                        cvPutText(centered,  buf, cvPoint(5, centered->height - 10), &font, CV_RGB(255,0,0));

                        sprintf(buf, "X = %4d", -(centerPad.y - (centered->height)/2));
                        cvPutText(centered,  buf, cvPoint(centered -> width - 130, centered->height - 30), &font, CV_RGB(0,255,0));
                        sprintf(buf, "Y = %4d", centerPad.x-(centered->width)/2);
                        cvPutText(centered,  buf, cvPoint(centered -> width - 130, centered->height - 10), &font, CV_RGB(0,255,0));

                        timeval currentTime;
                        gettimeofday(&currentTime, NULL);
                        int currentSecTime = currentTime.tv_sec - startTime.tv_sec;
                        int currentMilliSecTime = currentTime.tv_usec / 1000 - startTime.tv_usec / 1000;
                        double time = (currentSecTime * 1000) + currentMilliSecTime;

                        sprintf(buf, "Time %.2f sec", double(time)/1000);

                        cvPutText(centered,  buf, cvPoint(5, centered->height - 30), &font, CV_RGB(0,255,0));
                        cvPutText(centered,  "Looking for Orange", cvPoint(5, 25), &font, CV_RGB(0,255,0));

                        cvLine(centered, cvPoint(centered->width/2 - 10,centered->height/2),cvPoint(centered->width/2 + 10,centered->height/2), CV_RGB(0,255,0), 2, 8, 0);
                        cvLine(centered, cvPoint(centered->width/2,centered->height/2 - 10),cvPoint(centered->width/2,centered->height/2 + 10), CV_RGB(0,255,0), 2, 8, 0);

                        cvShowImage("With Center",centered);
                        cvShowImage("normalized", normalized);


                        cvReleaseImage(& normalized);
                        cvReleaseImage(& binarized_1);
                        cvReleaseImage(& centered);
                        cvReleaseImage(& filtered);
                        cvReleaseImage(& contours);
                        cvReleaseImageHeader(& image);


                        char c = cvWaitKey(33);
                        if (c == 27) { // МЮФЮРЮ ESC
                                break;
                        }
                        if (c == 109) { // МЮФЮРЮ л
                                flManHSV = true;
                        }
                        else flManHSV = false;

                        if ((cir_point_num < 3 && flManMode == false) || c == 120){ // ЙКЮБХЬЮ У
                                mode = WHITE_SQ;
                                goto start;
                        }
                }

        // НЯБНАНФДЮЕЛ ПЕЯСПЯШ
        cvReleaseCapture( &capture );

        // СДЮКЪЕЛ НЙМН
        cvDestroyWindow("resized");
        cvDestroyWindow("normalized");
        }

return 0;
}

