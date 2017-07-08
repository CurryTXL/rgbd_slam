#include "rgb_d_slam.h"
using namespace rgbd;
#include "parameter_reader.h"

// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int16.h>

// octomap
#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>
using namespace octomap;

class DroneExplorer
{
public:
  DroneExplorer();
  void initialize();

private:
  /**localizing and mapping based on rgbd slam*/

  vector<FRAME> m_keyframes;
  vector<FRAME> m_frames_to_be_map;

  g2o::SparseOptimizer m_global_optimizer;

  pcl::VoxelGrid<PointT> m_voxel_filter;
  pcl::PassThrough<PointT> m_grid_filter;
  pcl::StatisticalOutlierRemoval<PointT> m_outliers_filters;

  PointCloud::Ptr m_global_pointcloud;
  octomap::ColorOcTree *m_global_octree;
  bool m_need_to_save_map;

  string m_detector;
  string m_descriptor;
  CAMERA_INTRINSIC_PARAMETERS m_camera;
  double m_keyframe_threshold;
  bool m_check_loop_closure;
  int m_optimization_frames_number;
  double m_tree_resolution;
  double m_neighbor_radius;
  double m_gridsize;
  double m_mean_k;
  double m_std_dev;
  int m_start_index;

  ParameterReader m_parameter_reader;

public:
  void setStartIndex(int n);
  void processNewFrame(rgbd::FRAME frame);
  void computeKeyPointsAndDesp(rgbd::FRAME &frame);
  void addNewKeyFrame(rgbd::FRAME frame);
  bool mapNeedToUpdate();
  void optimizePoseGraph();
  void updateMap();
  void savePointCloudMap(std::string filename);
  void savePointCloudMap(PointCloud::Ptr pointcloud,
                         std::string filename);

private:
  /**exploration based on frontier*/
  octomap::point3d_collection m_current_frontiers;

public:
  void saveOctreeMap(std::string filename);
  void saveOctreeMap(octomap::ColorOcTree tree, std::string filename);
  void loadOctreeMap(std::string filename);
  void updateFrontierPoints();
  void visualizeFrontier();
  void visualizeCoordinateSystem();

  void searchTargetFrontier();
  PointCloud::Ptr convertFrontierToPointCloud();
  pcl::PointCloud<pcl::Normal>::Ptr
  estimateNormals(PointCloud::Ptr frontier_cloud);
  void filterUpAndDownwardNormals(
      pcl::PointCloud<pcl::Normal>::Ptr froniter_normals,
      pcl::PointCloud<PointT>::Ptr frontier_cloud,
      pcl::PointCloud<pcl::Normal>::Ptr filtered_normals,
      pcl::PointCloud<PointT>::Ptr filtered_cloud);
  void clusterNormals(
      pcl::PointCloud<pcl::Normal>::Ptr filtered_normals,
      pcl::PointCloud<PointT>::Ptr filtered_cloud,
      vector<pcl::PointCloud<pcl::Normal>::Ptr> &
          final_cluster_normals,
      vector<pcl::PointCloud<PointT>::Ptr> &final_cluster_cloud);
  void visualizeNormalsAndCloud(
      vector<pcl::PointCloud<pcl::Normal>::Ptr> final_cluster_normals,
      vector<pcl::PointCloud<PointT>::Ptr> final_cluster_cloud);
};
