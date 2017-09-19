//system and self
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

//vision process library
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/point_types_conversion.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/types.hpp>

//ros
#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int8.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/chain.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_ros/point_cloud.h>
#include <realsense_ros/camera_intrin.h>

//som(soft object manipulator related algorithm)
#include "TpMatching.h"


#define FPS 30
#define QUEUQ_MAX_SIZE 4

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, 
                                                        sensor_msgs::Image> syncPolicy;
cv_bridge::CvImagePtr imageRGB,imageDepth;

static cv::Rect2d boundingBox;
static std::vector<cv::Rect2d> boundingBoxs;
static bool paused;
static bool selectObject = false;
static bool startSelection = false;
//mouse callback
static void onMouse( int event, int x, int y, int, void* )
{
  if( !selectObject )
  {
    switch ( event )
    {
        case cv::EVENT_LBUTTONDOWN:
        //set origin of the bounding box
        startSelection = true;
        boundingBox.x = x;
        boundingBox.y = y;
        boundingBox.width = boundingBox.height = 0;
        break;
      case cv::EVENT_LBUTTONUP:
        //sei with and height of the bounding box
        boundingBox.width = std::abs( x - boundingBox.x );
        boundingBox.height = std::abs( y - boundingBox.y );
        paused = false;
        startSelection = false;
        selectObject = true;
        break;
      case cv::EVENT_MOUSEMOVE:

        if( startSelection && !selectObject )
        {
          //draw the bounding box
          cv::Mat currentFrame;
          imageRGB->image.copyTo( currentFrame );
          cv::rectangle( currentFrame, cv::Point((int) boundingBox.x, (int)boundingBox.y ), cv::Point( x, y ), cv::Scalar::all(255), 2, 1 );
          cv::imshow( "imageRGB", currentFrame );
        }
        break;
    }
  }
}
// void print_time(){
//     time_new = static_cast<double>( cv::getTickCount());
//     time = static_cast<double>((time_new) - time_old)/cv::getTickFrequency();
//     time_old = time_new;
//     std::cout<<"time:"<<time<<"\n";
// }
        
        
//depth image to colormap for visualize
cv::Mat getDepthColorMap(cv::Mat image_depth){
    double min;
    double max;
    cv::minMaxIdx(image_depth, &min, &max);
    cv::Mat adjMap;
    // expand your range to 0..255. Similar to histEq();
    float scale = 255 / (max-min);
    image_depth.convertTo(adjMap,CV_8UC1, scale, -min*scale); 

    // this is great. It converts your grayscale image into a tone-mapped one, 
    // much more pleasing for the eye
    // function is found in contrib module, so include contrib.hpp 
    // and link accordingly
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);
    return falseColorsMap;
}


//time to start tracking stage
bool received = false;
std::vector<cv::Mat> cls;
void chatterCallbackCVC(const sensor_msgs::ImageConstPtr& imgRGB,const sensor_msgs::ImageConstPtr& imgDepth){
    imageRGB = cv_bridge::toCvCopy(imgRGB,sensor_msgs::image_encodings::BGR8);
    imageDepth = cv_bridge::toCvCopy(imgDepth,sensor_msgs::image_encodings::TYPE_16UC1);
    received = true;
}

realsense_ros::camera_intrin camera_intrin_rgb;
void init_camera_info_rgb(){
    camera_intrin_rgb.ppx = 313.298400879;
    camera_intrin_rgb.ppy = 236.164962769;
    camera_intrin_rgb.fx = 617.998535156;
    camera_intrin_rgb.fy = 617.998596191;
    for(int i=0;i<5;i++)
        camera_intrin_rgb.coeffs.push_back(0);
    camera_intrin_rgb.dev_depth_scale = 0.000124986647279;
}
void camera_info_rgb_callback(const realsense_ros::camera_intrin camera_rgb){
    camera_intrin_rgb = camera_rgb;
}

realsense_ros::camera_intrin camera_intrin_depth;
void init_camera_info_depth(){
    camera_intrin_depth.ppx = 317.148406982;
    camera_intrin_depth.ppy = 246.048797607;
    camera_intrin_depth.fx = 475.876464844;
    camera_intrin_depth.fy = 475.876342773;
    double coeffs[] = {0.16355182230472565, -0.041357968002557755, 0.00481746019795537, 0.003345993347465992, 0.24256663024425507};
    for(int i=0;i<5;i++)
        camera_intrin_depth.coeffs.push_back(coeffs[i]);
    camera_intrin_depth.dev_depth_scale = 0.000124986647279;
}
void camera_info_depth_callback(const realsense_ros::camera_intrin camera_depth){
    camera_intrin_depth = camera_depth;
}

cv::Point3f deproject(cv::Point pixel, float depth_origin, realsense_ros::camera_intrin * intrin, bool is_depth = false){
    cv::Point3f point;
    float x = (pixel.x - intrin->ppx) / intrin->fx;
    float y = (pixel.y - intrin->ppy) / intrin->fy;
    if(is_depth)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    }
    double depth = depth_origin * intrin->dev_depth_scale;
    point.x = depth * x;
    point.y = depth * y;
    point.z = depth;
    return point;
}
cv::Point3f getContourCentroid(std::vector<cv::Point>& contour, cv::Point shift, cv::Mat image_depth, realsense_ros::camera_intrin * intrin,bool is_depth = false){
    cv::Point3f centroid;
    cv::Point centroid2d;
    double depth = 0;
    int valued_size = 0;
    std::vector<cv::Point> dilated_contour;
    for(int i=0;i<contour.size();i++)
    {
        centroid2d += contour[i];
    }
    centroid2d /= (int)contour.size();
    
    for(int i=0;i<contour.size();i++)
    {
        //manually dilate
        bool do_dilate = false;
        int dilate_max_size = 3;
        int dilate_min_size = 3;
        int x = contour[i].x-centroid2d.x;
        int y = contour[i].y-centroid2d.y;

//         if(x == 0 && y == 0){
//             std::cout<<"contour not closed\n";
//             for(int i=0;i<contour.size();i++)
//             {
//                 std::cout<<contour[i].x<<" "<<contour[i].y<<" ";
//             }
//             std::cout<<"\ndp:"<<(int)contour.size()<<"\n";
//             std::vector<std::vector<cv::Point> >cs;
//             cs.push_back(contour);
//             cv::drawContours( imageRGB->image, cs, 0, cv::Scalar::all(255), 1.6, 8,cv::noArray(),INT_MAX,shift);
//             cv::imshow("imageDepthDilated",imageRGB->image);
//             cv::waitKey(0);
//         }
        for(int j=dilate_min_size;j<=dilate_max_size;j++)
        {
            double dilate_scale = j/std::sqrt(x*x+y*y);
            cv::Point dilated_point;
            dilated_point.x = int(dilate_scale * x);
            dilated_point.y = int(dilate_scale * y);
            double depth_temp;
            if(do_dilate)
                depth_temp = image_depth.at<ushort>(contour[i] + dilated_point + shift);
            else
                depth_temp = image_depth.at<ushort>(contour[i] + shift);
            if(depth_temp > 0 && depth_temp < (1/intrin->dev_depth_scale))
            {
                depth += depth_temp;
                valued_size += 1;
                dilated_contour.push_back(contour[i] + dilated_point);
                break;
            }
        }
    }
    //show dilated_contour
//     contour = dilated_contour;
    if(valued_size == 0){
        std::cout<<"all contour points lose depth value ,contour size:"<<contour.size()<<"\n";
        cv::imwrite(ros::package::getPath("soft_object_tracking")+"/data/rgb.png",imageRGB->image);
        cv::imwrite(ros::package::getPath("soft_object_tracking")+"/data/depth.png",imageDepth->image);
        cv::Mat falseColorsMap = getDepthColorMap(image_depth);
        std::vector<std::vector<cv::Point> >cs;
        cs.push_back(dilated_contour);
        cv::drawContours( falseColorsMap, cs, 0, cv::Scalar::all(255), 1.6, 8,cv::noArray(),INT_MAX,shift);
        cv::imshow("imageDepthDilated",falseColorsMap);
        cv::waitKey(3);
        cv::waitKey(0);
    }
    depth /= valued_size;
    centroid2d += shift;
    centroid = deproject(centroid2d,depth,intrin);
    
    return centroid;
}
int main(int argc, char ** argv)
{
    ros::init(argc,argv, "soft_object_tracking_node", ros::init_options::NoSigintHandler);
    ros::NodeHandle node;
    std::string data_path = ros::package::getPath("soft_object_tracking")+"/data/tp.png";
    message_filters::Subscriber<sensor_msgs::Image> imageRGB_sub(node, "/camera/image/rgb_611205001943", 1000);
	message_filters::Subscriber<sensor_msgs::Image> imageDepth_sub(node, "/camera/image/registered_depth__611205001943", 1000);
	message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10),imageRGB_sub, imageDepth_sub);
	sync.registerCallback(boost::bind(&chatterCallbackCVC, _1, _2));
    
    ros::Subscriber subCameraInfoRGB = node.subscribe("/camera/camera_info/rgb_611205001943", 1, camera_info_rgb_callback);
    ros::Subscriber subCameraInfoDepth = node.subscribe("/camera/camera_info/depth_611205001943", 1, camera_info_depth_callback);
    
    ros::Publisher centroid_publisher = node.advertise<std_msgs::Float64MultiArray> ("/soft_object_tracking/centroid", 2);
    
    init_camera_info_rgb();
    init_camera_info_depth();
    //opencv
    cv::namedWindow("canny");
    cv::namedWindow("imageRGB");
    cv::namedWindow("imageDepth");
    cv::namedWindow("imageDepthDilated");
    cv::moveWindow("imageRGB",50,50);
    cv::moveWindow("imageDepth",50+640,50);
    cv::moveWindow("imageDepthDilated",50+640+640,50);
    cv::setMouseCallback( "imageRGB", onMouse, 0 );
    cv::RNG rng(12345);
    cv::Mat tp = cv::imread(data_path);
    som::TpMatching tp_matching;
    cv::MultiTracker trackers;

    cv::Mat imageShowRGB;
    cv::Mat imageShowDepth;
    bool initialized = false;
    std::deque<std::vector<cv::Point3f> >centroid_buffer;
    
    double time_new;
    double time_old = 0;
    double time;
    ros::Rate rate(FPS);
    while(ros::ok())
    {

        if(received)
        {
            imageRGB->image.copyTo(imageShowRGB);
            imageShowDepth = getDepthColorMap(imageDepth->image);
            if(selectObject)
            {
                selectObject = false;
                cv::Mat imageROI(imageRGB->image,boundingBox);
                cv::imwrite(data_path,imageROI);
            }
            if(!initialized && tp.data != NULL)
            {
                boundingBoxs = tp_matching.match(imageRGB->image,tp);
                for(int i=0;i<boundingBoxs.size();i++)
                    cv::rectangle(imageShowRGB,boundingBoxs[i],cv::Scalar::all(255),2,1);
                std::vector<cv::Ptr<cv::Tracker> > algorithms;
                for (int i = 0; i < boundingBoxs.size(); i++){
                    cv::TrackerKCF::Params kcfParams;
//                     kcfParams.detect_thresh = 0.08;
                    kcfParams.detect_thresh = 0.20;
                    kcfParams.max_patch_size=100*100;
                    
//                     cv::TrackerBoosting::Params boostParams;
//                     boostParams.numClassifiers = 100;
//                     boostParams.samplerOverlap = 2.0f;
//                     boostParams.samplerSearchFactor = 4.0f;
//                     boostParams.iterationInit = 50;
//                     boostParams.featureSetNumFeatures = ( boostParams.numClassifiers * 10 ) + boostParams.iterationInit;
                    
//                     cv::TrackerMIL::Params milParams;
//                     milParams.samplerInitInRadius = 3;
//                     milParams.samplerSearchWinSize = 25;
//                     milParams.samplerInitMaxNegNum = 65;
//                     milParams.samplerTrackInRadius = 4;
//                     milParams.samplerTrackMaxPosNum = 100000;
//                     milParams.samplerTrackMaxNegNum = 65;
//                     milParams.featureSetNumFeatures = 250;
                    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create(kcfParams);
                    algorithms.push_back(tracker);
                }
                //initializes the tracker
                if(!trackers.add(algorithms,imageRGB->image,boundingBoxs))
                {
                    std::cout << "***Could not initialize trackers...***\n";
                    return -1;
                }

                initialized = true;
            }
            else if(tp.data != NULL)
            {
                //updates the trackers
                if(trackers.update(imageRGB->image,boundingBoxs))
                {
                    std::vector<cv::Point3f> centroid_frame;
                    for(int i=0;i<boundingBoxs.size();i++)
                    {
                        int thresh_canny = 20;
                        cv::Mat canny_src_raw(imageRGB->image,boundingBoxs[i]);
                        std::vector<cv::Mat> channels;
                        cv::cvtColor(canny_src_raw,canny_src_raw,cv::COLOR_BGR2HSV);
                        cv::split(canny_src_raw, channels);
                        cv::Mat canny_src = channels[0];
                        cv::Mat canny_output;
                        std::vector<std::vector<cv::Point> > contours_origin;
                        std::vector<std::vector<cv::Point> > contours;
                        std::vector<cv::Vec4i> hierarchy;

                        /// Detect edges using canny
                        cv::Canny( canny_src, canny_output, thresh_canny, thresh_canny*2);
//                         cv::imshow("canny",canny_output);
//                         cv::waitKey(3);
                        /// Find contours_origin
                        cv::findContours( canny_output, contours_origin, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE );
                        /// Draw contours_origin
                        ///Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
                        double max_contour_area = -1;
                        int max_index = 0;
                        contours.resize(contours_origin.size());
                        for( int j = 0; j< contours_origin.size(); j++ )
                        {
                            cv::approxPolyDP(contours_origin[j],contours[j],0.1,true);
                            if(cv::contourArea(contours[j]) > max_contour_area)
                            {
                                max_contour_area = contourArea(contours[j]);
                                max_index = j;
                            }
                        }

                        if(contours.size() != 0)
                        {
                            //approximateDP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//                             std::vector<std::vector<cv::Point> >hull;
//                             hull.resize(1);
//                             cv::convexHull( contours[max_index], hull[0],true,true);
                            cv::Point shift = cv::Point(boundingBoxs[i].x,boundingBoxs[i].y);
//                             cv::rectangle(imageShowRGB,boundingBoxs[i],cv::Scalar::all(255),2,1);
//                             cv::drawContours( imageShowRGB, contours, max_index, cv::Scalar::all(255), 1.6, 8,cv::noArray(),INT_MAX,shift);
//                             cv::imshow("imageRGB",imageShowRGB);
//                             cv::waitKey(3);
                            cv::Point3f centroid = getContourCentroid(contours[max_index],shift,imageDepth->image,&camera_intrin_rgb);
                            centroid_frame.push_back(centroid);
                            cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                            ///Draw rectangle
                            cv::drawContours( imageShowRGB, contours, max_index, cv::Scalar::all(255), 1.6, 8,cv::noArray(),INT_MAX,shift);
                            cv::drawContours( imageShowDepth, contours, max_index, cv::Scalar::all(255), 2, 8,cv::noArray(),INT_MAX,shift);
                        }
                        else
                            std::cout<<"contours size is zero!!!!!"<<"\n";
                        cv::rectangle(imageShowRGB,boundingBoxs[i],cv::Scalar::all(255),2,1);
                    }
                    if(centroid_buffer.size() == QUEUQ_MAX_SIZE)
                        centroid_buffer.pop_front();
                    centroid_buffer.push_back(centroid_frame);
                    //use average value of buffer to smooth
                    std_msgs::Float64MultiArray centroid_smooth;
                    centroid_smooth.data.resize(centroid_frame.size()*3);
                    for(int i=0;i<centroid_smooth.data.size();i++)
                        centroid_smooth.data[i] = 0;
                    for(int i=0;i<centroid_buffer.size();i++)
                    {
                        for(int j=0;j<centroid_frame.size();j++)
                        {
                            centroid_smooth.data[0+j*3] += centroid_buffer[i][j].x;
                            centroid_smooth.data[1+j*3] += centroid_buffer[i][j].y;
                            centroid_smooth.data[2+j*3] += centroid_buffer[i][j].z; 
                        }
                        
                    }
                    for(int i=0;i<centroid_smooth.data.size();i++)
                        centroid_smooth.data[i] /= centroid_buffer.size();



//                         centroid_smooth.data[i] = 0;
                    //Publish
                    centroid_publisher.publish(centroid_smooth);
                }
            }
            cv::imshow("imageRGB",imageShowRGB);
            cv::imshow("imageDepth",imageShowDepth);
            cv::waitKey(3);
            received = false;
        }
        else
            std::cout<<"didn't receive image!!\n";
        
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}

//
// File trailer for soft_object_tracking_node.cpp
//
// [EOF]
//
