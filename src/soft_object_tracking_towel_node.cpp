//system and self
#include <iostream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

//vision process library
#include <pcl/visualization/cloud_viewer.h>
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
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

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

#define FPS 10

//return the centroid of given points
pcl::PointXYZRGB getCentroid(std::vector<pcl::PointXYZRGB> p)
{
    pcl::PointXYZRGB centroid;
    for(int i=0;i<p.size();i++){
        centroid.x += p[i].x;
        centroid.y += p[i].y;
        centroid.z += p[i].z;
    }
    centroid.x /= p.size();
    centroid.y /= p.size();
    centroid.z /= p.size();
    return centroid;
    
}
//return the distance between two points
double dist(pcl::PointXYZRGB p1,pcl::PointXYZRGB p2)
{
    return (std::sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z)));
}

//return value higher than threshold
void highPass(double *pointer,int size, double threshold)
{
    for(int i=0;i<size;i++){
        if(std::abs(pointer[i])<threshold)
            pointer[i] = 0;
    }
}

//return soomth value
void lowPass(double *pointer,pcl::PointXYZRGB centroid11, pcl::PointXYZRGB centroid22)
{
    double lambda = 0;
    pointer[0] = lambda*pointer[0] + (1-lambda)*centroid11.x;
    pointer[1] = lambda*pointer[1] + (1-lambda)*centroid11.y;
    pointer[2] = lambda*pointer[2] + (1-lambda)*centroid11.z;
    pointer[3] = lambda*pointer[3] + (1-lambda)*centroid22.x;
    pointer[4] = lambda*pointer[4] + (1-lambda)*centroid22.y;
    pointer[5] = lambda*pointer[5] + (1-lambda)*centroid22.z;
}



pcl::PointCloud<pcl::PointXYZRGB >::Ptr cldC(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB >::Ptr cldROIC(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");

//global variable feedback points
double p[6] = {0};
//time to start tracking stage
bool confirmed = false;
bool is_getCen = false;
pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true); // Initializing with true will allow us to extract the removed indices
pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
pcl::PointXYZRGB centroid11;
pcl::PointXYZRGB centroid22;
void pointCallback(const sensor_msgs::PointCloud2ConstPtr& pclPts){
    pcl::fromROSMsg<pcl::PointXYZRGB >(*pclPts,*cldC);
    
    pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud (cldC);

    pass.setFilterFieldName ("x");
    if(confirmed)
        pass.setFilterLimits (-0.3, 0.3);
    else
        pass.setFilterLimits (-0.1, 0.1);

	pass.filter (*cldROIC);

    
    pcl::PointXYZHSV temp;
    double xCenter = 0,yCenter = 0,zCenter = 0;
	for(int i=0;i<cldROIC->size();i++){
		pcl::PointXYZRGBtoXYZHSV(cldROIC->points[i],temp);
		if(temp.h>160 && temp.h<210 && temp.s>0.4){
			inliers->indices.push_back(i);
			xCenter +=cldROIC->points[i].x;
            
// 			yCenter +=cldROIC->points[i].y;
// 			zCenter +=cldROIC->points[i].z;
		}
	}
	xCenter /= inliers->indices.size();
	
	eifilter.setInputCloud (cldROIC);
	eifilter.setIndices (inliers);
	eifilter.filter (*cldROIC);
    inliers->indices.clear();
    
    
    std::vector<pcl::PointXYZRGB> selectedPoint1;
    std::vector<pcl::PointXYZRGB> selectedPoint2;
    std::vector<pcl::PointXYZRGB> selectedPoint11;
    std::vector<pcl::PointXYZRGB> selectedPoint22;
    for(int i=0;i<cldROIC->size();i++){
        if(confirmed){
            double dist1 = dist(cldROIC->points[i],centroid11);
            double dist2 = dist(cldROIC->points[i],centroid22);
            if(dist1 < std::min(dist2,0.02)){
                selectedPoint1.push_back(cldROIC->points[i]);
                cldROIC->points[i].g = 255;
            }
            else if(dist2 < std::min(dist1,0.02)){
                selectedPoint2.push_back(cldROIC->points[i]);
                cldROIC->points[i].g = 255;
            }
        }
        else{
            if(cldROIC->points[i].x < xCenter){
                selectedPoint1.push_back(cldROIC->points[i]);
            }
            else{
                selectedPoint2.push_back(cldROIC->points[i]);
            }
        }
        
    }
    
    
//     std::cout<<"size1: "<<sizes[0]<<"  size2:"<<sizes[1]<<std::endl;
    if(selectedPoint1.size() > 5 && selectedPoint2.size() > 5){
        pcl::PointXYZRGB centroid1,centroid2;
        //if confirmed, use last centroid to estimate current centroid.Otherwise compute the centroid use all points
        
        centroid1 = getCentroid(selectedPoint1);
        centroid2 = getCentroid(selectedPoint2);
        if(confirmed){
            selectedPoint11 = selectedPoint1;
            selectedPoint22 = selectedPoint2;
        }
        else{
            for(int i=0;i<selectedPoint1.size();i++){
                if(dist(selectedPoint1[i],centroid1)<0.1)
                    selectedPoint11.push_back(selectedPoint1[i]);
                }
            for(int i=0;i<selectedPoint2.size();i++){
                if(dist(selectedPoint2[i],centroid2)<0.1)
                    selectedPoint22.push_back(selectedPoint2[i]);
                }
        }
        if(selectedPoint11.size() > 5 && selectedPoint22.size() > 5){
            centroid11 = getCentroid(selectedPoint11);
            centroid22 = getCentroid(selectedPoint22);
            centroid11.r = 255;
            centroid22.r = 255;
            
            //high pass to filter noise
//             highPass(p,6,1e-3);
            //low pass to filter noise
            lowPass(p,centroid11,centroid22);
            cldROIC->points.push_back(centroid11);
            cldROIC->points.push_back(centroid22);
            
            confirmed = true;
            is_getCen = true;
        }
    }
    
	viewer.showCloud(cldROIC);

}


int main(int argc, char ** argv)
{
    ros::init(argc,argv, "soft_object_tracking_towel_node", ros::init_options::NoSigintHandler);
    ros::NodeHandle node;
    ros::Subscriber imagePoint_sub = node.subscribe("/camera/depth_registered/points_SR300_611205001943", 2, pointCallback);
    ros::Publisher centroid_publisher = node.advertise<std_msgs::Float64MultiArray> ("/soft_object_tracking/centroid", 2);
    

    ros::Rate rate(FPS);
    while(ros::ok())
    {
        if(viewer.wasStopped())
            break;
        if(is_getCen)
        {
            std_msgs::Float64MultiArray centroid_frame;
            for(int i=0;i<6;i++)
                centroid_frame.data.push_back(p[i]);
            centroid_publisher.publish(centroid_frame);
        }
        else
            std::cout<<"Didn't receive the feature points\n";

        ros::spinOnce();
        rate.sleep();
    }
            

}
