#include <ros/ros.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZRGB>);
  pcl::fromROSMsg(*cloud_msg, *cloud);
  pcl::io::savePCDFileASCII (ros::package::getPath("soft_object_tracking_node")+"/data/target.pcd", *cloud);
  std::cout << "Saved " << cloud->size () << " data points to target.pcd." << std::endl;
}

int main (int argc, char** argv)
{

  // Initialize ROS
  ros::init (argc, argv, "soft_object_tracking_record_node");

  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points_SR300_611205001943", 1, cloud_cb);
  // Create a ROS publisher for the output point cloud
  // Spin
  ros::spin();
  return 0;
}
