#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;
namespace som{
class TpMatching{
private:
    int mechod = CV_TM_CCOEFF_NORMED;
    double thresh = 0.7;
    cv::Mat result;
    int dist(Rect2d p1, Point p2)
    {
        return ( (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) );
    }
public:
    TpMatching(){};
    TpMatching(int mechod,double thresh)
    {
        this->mechod = mechod;
        this->thresh = thresh;
    }
    std::vector<cv::Rect2d> match(cv::Mat src, cv::Mat tp)
    {
        /// Create the result matrix
        int result_cols =  src.cols - tp.cols + 1;
        int result_rows = src.rows - tp.rows + 1;

        result.create( result_cols, result_rows, CV_32FC1 );

        /// Do the Matching and Normalize
        matchTemplate( src, tp, result, mechod );
        normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
        threshold(result, result, thresh, 1., THRESH_TOZERO);
        
        Mat1b resb;
        result.convertTo(resb, CV_8U, 255);

        vector<vector<cv::Point> > contours;
        findContours(resb, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        std::vector<cv::Rect2d> boundingBox;
        for (int i=0; i<contours.size(); ++i)
        {
            Mat1b mask(result.rows, result.cols, uchar(0));
            drawContours(mask, contours, i, Scalar(255), CV_FILLED);
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            Point matchLoc;
            /// Localizing the best match with minMaxLoc
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, mask);
            matchLoc = maxLoc;
            int j=0;
            for(;j<boundingBox.size();j++)
            {
                if (dist(boundingBox[j],matchLoc) < 1)
                    break;
            }
            if(j == boundingBox.size())
                boundingBox.push_back(cv::Rect2d(matchLoc.x,matchLoc.y,tp.cols,tp.rows));
        }
        return boundingBox;
    }
    virtual ~TpMatching(){};
};
}   //namespace som
