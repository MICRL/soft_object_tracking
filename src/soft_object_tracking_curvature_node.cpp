#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/PointCloud2.h>

#include <vector>
#include <algorithm>
#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types_conversion.h>

#include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>

#include <pcl/search/kdtree.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>

#include <pcl_conversions/pcl_conversions.h>

#define TARGET 1 // NOTE: If you do not want to show the target, change this to 0

#define RADIUS 0.015f // 0.03
#define K 10 // 10
#define TOP 10 // 10
#define DOWN_SAMPLE_SIZE 0.01f // 0.01

#if (TARGET)
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB target_centroid;
  pcl::PointXYZRGB target_end_point;
  void process_target(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud);
#endif

ros::Publisher filtered_pub;
ros::Publisher downsampled_pub;
ros::Publisher feature_publisher;
ros::Publisher max_cur_pub;


boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

//return the centroid of given points
pcl::PointXYZRGB getCentroid(pcl::PointCloud <pcl::PointXYZRGB> p)
{
  pcl::PointXYZRGB centroid;
  for(int i=0;i<p.size();i++)
  {
    centroid.x += p[i].x;
    centroid.y += p[i].y;
    centroid.z += p[i].z;
  }
  centroid.x /= p.size();
  centroid.y /= p.size();
  centroid.z /= p.size();
  return centroid;
}

void cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  // Init variables
  pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true); // Initializing with true will allow us to extract the removed indices
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZRGB>);
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud <pcl::PointXYZRGB>);
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud_downsampled (new pcl::PointCloud <pcl::PointXYZRGB>);

  // Read the cloud from the ROS msg
  pcl::fromROSMsg(*cloud_msg, *cloud);


  pcl::VoxelGrid<pcl::PointXYZRGB> ds;
  ds.setInputCloud (cloud);
  ds.setLeafSize (DOWN_SAMPLE_SIZE, DOWN_SAMPLE_SIZE, DOWN_SAMPLE_SIZE);
  ds.filter (*cloud_downsampled);

  // std::cout << cloud_filtered->size() << std::endl;

  // Cut the depth of the cloud
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (cloud_downsampled);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 2.0);

  pass.filter (*cloud_filtered);

  // Extract the points with the aprropriate color
  pcl::PointXYZHSV temp;
  for(int i=0;i<cloud_filtered->size();i++)
  {
		pcl::PointXYZRGBtoXYZHSV(cloud_filtered->points[i],temp);
		if(temp.h>160 && temp.h<210 && temp.s>0.6)
    {
			inliers->indices.push_back(i);
      cloud_filtered->points[i].g = 255;
		}
	}

  eifilter.setInputCloud (cloud_filtered);
	eifilter.setIndices (inliers);
	eifilter.filter (*cloud_filtered);

  // Eliminate the outlier if the cloud is large enough
  if (cloud_filtered->size() > 0)
  {
    std_msgs::Float64MultiArray features;
    features.data.resize(4);

    // Calculate the curvature
    if (cloud_filtered->size() > 0)
    {
      //for normal computing
      pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setInputCloud (cloud_filtered);
      // ne.setSearchSurface (cloud_filtered);
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
      ne.setSearchMethod (tree);
      ne.setKSearch (K);
      // ne.setRadiusSearch (RADIUS);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);
      ne.compute (*cloud_with_normals);

      #if (USETIME)
        gettimeofday(&end, NULL);
        seconds  = end.tv_sec  - start.tv_sec;
        useconds = end.tv_usec - start.tv_usec;
        mtime = ((seconds) * 1000 + useconds/1000.0) + 0.5;
        normal_calculation = mtime;
        gettimeofday(&start, NULL);
      #endif

      // Setup the principal curvatures computation
      pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;

      // Provide the original point cloud (without normals)
      principal_curvatures_estimation.setInputCloud (cloud_filtered);

      // Provide the point cloud with normals
      principal_curvatures_estimation.setInputNormals (cloud_with_normals);

      // Use the same KdTree from the normal estimation
      principal_curvatures_estimation.setSearchMethod (tree);
      principal_curvatures_estimation.setRadiusSearch (1.0);

      // Actually compute the principal curvatures
      pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
      principal_curvatures_estimation.compute (*principal_curvatures);

      //find the max curvature and show its principal direction
      // float sum_curvature = 0;
      float max_curvature = 0;
      int distance = 0;
      // std::vector<float> *valid_curvatures = new std::vector<float>();
      for(int i=0;i<principal_curvatures->size();i++)
      {
        float curvature = principal_curvatures->points[i].pc1;
        if (!std::isnan(curvature))
        {
          if (curvature > max_curvature)
          {
            max_curvature = curvature;
            distance = i;
          }
        }
      }

      if (distance > 0)
      {
        pcl::PointXYZRGB centroid_3D = getCentroid(*cloud_filtered);

        features.data[0] = centroid_3D.x;
        features.data[1] = centroid_3D.y;
        features.data[2] = centroid_3D.z;
        features.data[3] = max_curvature;

        feature_publisher.publish (features);

        //
        std_msgs::Float32 max_cur;
        max_cur.data = max_curvature;
        max_cur_pub.publish (max_cur);

        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        viewer->setBackgroundColor (0, 0, 0);

        cloud_filtered->push_back(centroid_3D);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_filtered);
        viewer->addPointCloud<pcl::PointXYZRGB> (cloud_filtered, rgb, "sample cloud");

        #if (TARGET)
          pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> target_rgb(target_cloud);
          viewer->addPointCloud<pcl::PointXYZRGB> (target_cloud, target_rgb, "target cloud");
        #endif

        viewer->spinOnce (10);

      }
      else
      {
        std::cout << "No valid points!" << std::endl << std::endl;
      }
    }
  }

  // Convert to ROS data type
  sensor_msgs::PointCloud2 filtered_output;
  sensor_msgs::PointCloud2 downsampled_output;

  pcl::toROSMsg(*cloud_filtered, filtered_output);
  pcl::toROSMsg(*cloud_downsampled, downsampled_output);

  filtered_output.header = cloud_msg->header;
  downsampled_output.header = cloud_msg->header;

  // Publish the data
  filtered_pub.publish (filtered_output);
  downsampled_pub.publish (downsampled_output);
}

int main (int argc, char** argv)
{

  // Initialize ROS
  ros::init (argc, argv, "soft_object_tracking_curvature_node");
  ros::NodeHandle nh;

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/depth_registered/points_SR300_611205001943", 1, cloud_cb);
  feature_publisher = nh.advertise<std_msgs::Float64MultiArray> ("/soft_object_tracking/centroid", 2);

  // Create a ROS publisher for the output point cloud
  filtered_pub = nh.advertise<sensor_msgs::PointCloud2> ("filtered_output", 1);
  downsampled_pub = nh.advertise<sensor_msgs::PointCloud2> ("downsampled_output", 1);
  max_cur_pub = nh.advertise<std_msgs::Float32> ("max_cur", 1);

  #if (TARGET)
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (ros::package::getPath("soft_object_tracking")+"/data/target.pcd", *target_cloud) == -1) //* load the file
    {
      PCL_ERROR ("Couldn't read file target.pcd \n");
      return (-1);
    }
    process_target(target_cloud);
  #endif

  // Spin
  ros::spin ();

  return 0;
}

#if (TARGET)
  void process_target(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target_cloud)
  {
    // Init variables
    pcl::ExtractIndices<pcl::PointXYZRGB> eifilter (true); // Initializing with true will allow us to extract the removed indices
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    pcl::VoxelGrid<pcl::PointXYZRGB> ds;
    ds.setInputCloud (target_cloud);
    ds.setLeafSize (DOWN_SAMPLE_SIZE, DOWN_SAMPLE_SIZE, DOWN_SAMPLE_SIZE);
    ds.filter (*target_cloud);

    // Cut the depth of the cloud
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (target_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 2.0);

    pass.filter (*target_cloud);
    // Extract the points with the aprropriate color
    pcl::PointXYZHSV temp;
    for(int i=0;i<target_cloud->size();i++)
    {
  		pcl::PointXYZRGBtoXYZHSV(target_cloud->points[i],temp);
  		if(temp.h>160 && temp.h<210 && temp.s>0.6)
      {
  			inliers->indices.push_back(i);
        target_cloud->points[i].r = 255;
        target_cloud->points[i].g = 70;
        target_cloud->points[i].b = 70;
  		}
  	}

    eifilter.setInputCloud (target_cloud);
  	eifilter.setIndices (inliers);
  	eifilter.filter (*target_cloud);

    // Eliminate the outlier if the cloud is large enough
    if (target_cloud->size() > 100)
    {
    // Calculate the curvature
      //for normal computing
      pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setInputCloud (target_cloud);
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
      ne.setSearchMethod (tree);
      ne.setKSearch (K);
      // ne.setRadiusSearch (RADIUS);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);
      ne.compute (*cloud_with_normals);

      pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> principal_curvatures_estimation;
      principal_curvatures_estimation.setInputCloud (target_cloud);
      principal_curvatures_estimation.setInputNormals (cloud_with_normals);
      principal_curvatures_estimation.setSearchMethod (tree);
      principal_curvatures_estimation.setRadiusSearch (1.0);

      // Actually compute the principal curvatures
      pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principal_curvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
      principal_curvatures_estimation.compute (*principal_curvatures);

      float max_curvature = 0;
      int distance = 0;
      for(int i=0;i<principal_curvatures->size();i++)
      {
        float curvature = principal_curvatures->points[i].pc1;
        if (!std::isnan(curvature))
        {
          if (curvature > max_curvature)
          {
            max_curvature = curvature;
            distance = i;
          }
        }
      }

      if (distance >= 0)
      {
        target_centroid = getCentroid(*target_cloud);
        target_end_point = pcl::PointXYZRGB(target_centroid);

        target_centroid.r = 255;
        target_centroid.g = 255;

        target_cloud->push_back(target_centroid);

      }
      else
      {
        std::cout << "Target Has No valid points!" << std::endl << std::endl;
      }
    }
  }
#endif
