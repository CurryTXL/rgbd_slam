#include "drone_explorer.h"

DroneExplorer::DroneExplorer()
{
}

void DroneExplorer::setStartIndex(int n)
{
  m_start_index= n;
}

void DroneExplorer::computeKeyPointsAndDesp(rgbd::FRAME &frame)
{
  rgbd::computeKeyPointsAndDesp(frame, m_detector, m_descriptor);
}

void DroneExplorer::initialize()
{
  /*initialize parameter*/
  m_detector= m_parameter_reader.getData("detector");
  m_descriptor= m_parameter_reader.getData("descriptor");
  m_camera= rgbd::getDefaultCamera();
  m_keyframe_threshold=
      atof(m_parameter_reader.getData("keyframe_threshold").c_str());
  m_check_loop_closure= m_parameter_reader.getData(
                            "check_loop_closure") == string("yes");
  m_optimization_frames_number=
      atoi(m_parameter_reader.getData("optimization_frames_number")
               .c_str());
  m_tree_resolution=
      atof(m_parameter_reader.getData("tree_resolution").c_str());
  m_neighbor_radius=
      atof(m_parameter_reader.getData("neighbor_radius").c_str());
  m_gridsize= atof(m_parameter_reader.getData("voxel_grid").c_str());
  m_mean_k= atoi(m_parameter_reader.getData("mean_k").c_str());
  m_std_dev= atof(m_parameter_reader.getData("std_dev").c_str());

  /*initialize point cloud filters*/
  m_voxel_filter.setLeafSize(m_gridsize, m_gridsize, m_gridsize);

  m_grid_filter.setFilterFieldName("z");
  m_grid_filter.setFilterLimits(0.0, 4.0);

  m_outliers_filters.setMeanK(m_mean_k);
  m_outliers_filters.setStddevMulThresh(m_std_dev);

  /*initialize global map*/
  m_global_pointcloud=
      pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
  m_global_octree= new octomap::ColorOcTree(m_tree_resolution);

  /*initialize g2o optimizer*/
  SlamLinearSolver *linearSolver= new SlamLinearSolver();
  linearSolver->setBlockOrdering(false);
  SlamBlockSolver *blockSolver= new SlamBlockSolver(linearSolver);
  g2o::OptimizationAlgorithmLevenberg *solver=
      new g2o::OptimizationAlgorithmLevenberg(blockSolver);

  m_global_optimizer.setAlgorithm(solver);
  m_global_optimizer.setVerbose(false);

  g2o::VertexSE3 *v= new g2o::VertexSE3();
  v->setId(m_start_index);
  v->setEstimate(Eigen::Isometry3d::Identity());
  v->setFixed(true);
  m_global_optimizer.addVertex(v);
}

/**main process of rgbd slam*/
void DroneExplorer::processNewFrame(rgbd::FRAME frame)
{
  rgbd::computeKeyPointsAndDesp(frame, m_detector, m_descriptor);

  rgbd::CHECK_RESULT result= rgbd::checkKeyframes(
      m_keyframes.back(), frame, m_global_optimizer);

  if(result == KEYFRAME)
  {
    cout << "This is a new keyframe" << endl;
    addNewKeyFrame(frame);

    if(m_check_loop_closure)
    {
      rgbd::checkNearbyLoops(m_keyframes, frame, m_global_optimizer);
      rgbd::checkRandomLoops(m_keyframes, frame, m_global_optimizer);
    }

    /*optimize when have enough new frame*/
    if(mapNeedToUpdate())
    {
      optimizePoseGraph();
      updateMap();
    }
  }
}

void DroneExplorer::addNewKeyFrame(rgbd::FRAME frame)
{
  m_keyframes.push_back(frame);
  m_frames_to_be_map.push_back(frame);
}

bool DroneExplorer::mapNeedToUpdate()
{
  if(m_frames_to_be_map.size() > m_optimization_frames_number)
    return true;
  else
    return false;
}

void DroneExplorer::optimizePoseGraph()
{
  cout << "optimizing pose graph, vertices: "
       << m_global_optimizer.vertices().size() << endl;
  m_global_optimizer.initializeOptimization();
  m_global_optimizer.optimize(100);
}

/**update point cloud map and octree map*/
void DroneExplorer::updateMap()
{
  PointCloud::Ptr local_pointcloud_map(new PointCloud());
  PointCloud::Ptr transformed_pointcloud(new PointCloud());
  // point cloud filtered
  PointCloud::Ptr filtered_pointcloud(new PointCloud());
  for(size_t i= 0; i < m_frames_to_be_map.size(); i++)
  {
    /*update point cloud map*/
    g2o::VertexSE3 *vertex= dynamic_cast<g2o::VertexSE3 *>(
        m_global_optimizer.vertex(m_frames_to_be_map[i].frameID));
    Eigen::Isometry3d pose= vertex->estimate();
    PointCloud::Ptr newCloud=
        rgbd::image2PointCloud(m_frames_to_be_map[i].rgb,
                               m_frames_to_be_map[i].depth, m_camera);

    m_voxel_filter.setInputCloud(newCloud);
    m_voxel_filter.filter(*filtered_pointcloud);
    m_grid_filter.setInputCloud(filtered_pointcloud);
    m_grid_filter.filter(*newCloud);

    pcl::transformPointCloud(*newCloud, *transformed_pointcloud,
                             pose.matrix());
    *local_pointcloud_map+= *transformed_pointcloud;
    transformed_pointcloud->clear();
    newCloud->clear();
    /*filter point cloud*/
    m_voxel_filter.setInputCloud(local_pointcloud_map);
    m_voxel_filter.filter(*filtered_pointcloud);
    m_outliers_filters.setInputCloud(filtered_pointcloud);
    m_outliers_filters.filter(*local_pointcloud_map);
    /*add to global point cloud*/
    *m_global_pointcloud+= *local_pointcloud_map;
    cout << "point cloud update done" << endl;

    /*update octree map*/
    for(auto p : local_pointcloud_map->points)
    {
      // use  insertRay to update free points
      point3d cp(m_frames_to_be_map.at(i).cameraPos[0],
                 m_frames_to_be_map.at(i).cameraPos[1],
                 m_frames_to_be_map.at(i).cameraPos[2]);
      m_global_octree->insertRay(cp, octomap::point3d(p.x, p.y, p.z));
    }
    for(auto p : local_pointcloud_map->points)
    {
      // update occupied points
      ColorOcTreeNode *node=
          m_global_octree->search(point3d(p.x, p.y, p.z));
      node->setLogOdds(2.2);
    }
    for(auto p : local_pointcloud_map->points)
    {
      m_global_octree->integrateNodeColor(p.x, p.y, p.z, p.r, p.g,
                                          p.b);
    }
    m_global_octree->updateInnerOccupancy();
    cout << "octomap update done." << endl;
    local_pointcloud_map->clear();
  }
  m_frames_to_be_map.clear();
}

void DroneExplorer::savePointCloudMap(std::string filename)
{
  pcl::io::savePCDFile(filename, *m_global_pointcloud);
}

void DroneExplorer::savePointCloudMap(PointCloud::Ptr pointcloud,
                                      std::string filename)
{
  pcl::io::savePCDFile(filename, *pointcloud);
}

void DroneExplorer::saveOctreeMap(octomap::ColorOcTree tree,
                                  std::string filename)
{
  tree.write(filename);
}

void DroneExplorer::saveOctreeMap(std::string filename)
{
  m_global_octree->write(filename);
}

void DroneExplorer::visualizeFrontier()
{
  for(int i= 0; i < m_current_frontiers.size(); i++)
  {
    point3d pt= m_current_frontiers.at(i);
    m_global_octree->updateNode(pt, true);
    m_global_octree->integrateNodeColor(pt.x(), pt.y(), pt.z(), 255,
                                        255, 255);
    m_global_octree->setNodeColor(pt.x(), pt.y(), pt.z(), 255, 255,
                                  255);
  }
  m_global_octree->updateInnerOccupancy();
}

void DroneExplorer::visualizeCoordinateSystem()
{
  m_global_octree->updateNode(point3d(0, 0, 0), true);
  m_global_octree->integrateNodeColor(0, 0, 0, 255, 255, 255);
  for(int i= 1; i <= 20; i++)
  {
    m_global_octree->updateNode(point3d(m_tree_resolution * i, 0, 0),
                                true);
    m_global_octree->integrateNodeColor(m_tree_resolution * i, 0, 0,
                                        255, 0, 0);
    m_global_octree->updateNode(point3d(0, m_tree_resolution * i, 0),
                                true);
    m_global_octree->integrateNodeColor(0, m_tree_resolution * i, 0,
                                        0, 255, 0);
    m_global_octree->updateNode(point3d(0, 0, m_tree_resolution * i),
                                true);
    m_global_octree->integrateNodeColor(0, 0, m_tree_resolution * i,
                                        0, 0, 255);
  }
  m_global_octree->updateInnerOccupancy();
}

/**search a target at current frontier*/
void DroneExplorer::searchTargetFrontier()
{
  /*convert frontier to pcl point cloud*/
  PointCloud::Ptr frontier_cloud= convertFrontierToPointCloud();

  /*estimate normals*/
  pcl::PointCloud<pcl::Normal>::Ptr frontier_normals=
      estimateNormals(frontier_cloud);

  /*filter some normals*/
  pcl::PointCloud<pcl::Normal>::Ptr filtered_normals(
      new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<PointT>::Ptr filtered_cloud(
      new pcl::PointCloud<PointT>);
  filterUpAndDownwardNormals(frontier_normals, frontier_cloud,
                             filtered_normals, filtered_cloud);

  // /*cluster normals using kmeans*/
  vector<pcl::PointCloud<pcl::Normal>::Ptr> final_cluster_normals;
  vector<pcl::PointCloud<PointT>::Ptr> final_cluster_cloud;
  clusterNormals(filtered_normals, filtered_cloud,
                 final_cluster_normals, final_cluster_cloud);

  /*visualize cloud and normals*/
  visualizeNormalsAndCloud(final_cluster_normals,
                           final_cluster_cloud);
}

void DroneExplorer::loadOctreeMap(std::string filename)
{
  AbstractOcTree *tree= AbstractOcTree::read(filename);
  m_global_octree= dynamic_cast<ColorOcTree *>(tree);
}

/**update frontier by searching octree map*/
void DroneExplorer::updateFrontierPoints()
{
  for(ColorOcTree::leaf_iterator it= m_global_octree->begin_leafs(),
                                 end= m_global_octree->end_leafs();
      it != end; ++it)
  {
    /*skip no-free points*/
    double occupancy= it->getOccupancy();
    if(occupancy > 0.75)
      continue;
    ROS_INFO_STREAM("free frontier points," << m_tree_resolution);
    point3d node= it.getCoordinate();
    for(int i= -1; i <= 1; i++)
    {
      for(int j= -1; j <= 1; j++)
      {
        for(int k= -1; k <= 1; k++)
        {
          if(i == 0 && j == 0 && k == 0)
            continue;
          point3d pt(node.x() + i * m_tree_resolution,
                     node.y() + j * m_tree_resolution,
                     node.z() + k * m_tree_resolution);
          ColorOcTreeNode *near_node= m_global_octree->search(pt);
          if(near_node != NULL)
          {
          }
          else
          {
            // add a frontier node
            m_current_frontiers.push_back(pt);
          }
        }
      }
    }
  }
  ROS_INFO_STREAM(
      "current frontier size:" << m_current_frontiers.size());
}

/*create point cloud from octomap frontier*/
PointCloud::Ptr DroneExplorer::convertFrontierToPointCloud()
{
  ROS_INFO_STREAM("start converting frontier");
  PointCloud::Ptr frontier_cloud(new PointCloud);
  for(int i= 0; i < m_current_frontiers.size(); i++)
  {
    point3d pt= m_current_frontiers.at(i);
    PointT p;
    p.x= pt.x();
    p.y= pt.y();
    p.z= pt.z();
    p.r= p.g= p.g= 255;
    frontier_cloud->points.push_back(p);
  }
  frontier_cloud->height= 1;
  frontier_cloud->width= frontier_cloud->points.size();
  frontier_cloud->is_dense= false;

  return frontier_cloud;
}

/*estimate normals using pcl */
pcl::PointCloud<pcl::Normal>::Ptr
DroneExplorer::estimateNormals(PointCloud::Ptr frontier_cloud)
{
  pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
  normal_estimator.setInputCloud(frontier_cloud);

  pcl::search::KdTree<PointT>::Ptr kdTree(
      new pcl::search::KdTree<PointT>());
  normal_estimator.setSearchMethod(kdTree);

  pcl::PointCloud<pcl::Normal>::Ptr froniter_normals(
      new pcl::PointCloud<pcl::Normal>);
  /*Use all neighbors in a sphere of radius 50cm*/
  normal_estimator.setRadiusSearch(m_neighbor_radius);
  normal_estimator.compute(*froniter_normals);

  return froniter_normals;
}

/*filter upward and downward normals*/
void DroneExplorer::filterUpAndDownwardNormals(
    pcl::PointCloud<pcl::Normal>::Ptr froniter_normals,
    pcl::PointCloud<PointT>::Ptr frontier_cloud,
    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals,
    pcl::PointCloud<PointT>::Ptr filtered_cloud)
{
  double unit_normal_z[3]= { 0, 1.0, 0 };
  vector<int> normal_indices;
  for(size_t i= 0; i < froniter_normals->size(); i++)
  {
    double dot_product=
        unit_normal_z[0] * froniter_normals->at(i).normal_x +
        unit_normal_z[1] * froniter_normals->at(i).normal_y +
        unit_normal_z[2] * froniter_normals->at(i).normal_z;
    if(fabs(dot_product) < 0.5)
      normal_indices.push_back(i);
  }

  for(size_t i= 0; i < normal_indices.size(); i++)
  {
    filtered_normals->push_back(
        froniter_normals->at(normal_indices.at(i)));
    filtered_cloud->push_back(
        frontier_cloud->at(normal_indices.at(i)));
  }
  ROS_INFO_STREAM("filter normal:" << filtered_normals->size());
  ROS_INFO_STREAM("filter cloud:" << filtered_cloud->size());
}

/**use kmeans cluster to classify different surface*/
void DroneExplorer::clusterNormals(
    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals,
    pcl::PointCloud<PointT>::Ptr filtered_cloud,
    vector<pcl::PointCloud<pcl::Normal>::Ptr> &final_cluster_normals,
    vector<pcl::PointCloud<PointT>::Ptr> &final_cluster_cloud)
{
  int cluster_count= 4;
  cv::Mat cv_filtered_normals(filtered_normals->size(), 3, CV_32F);
  cv::Mat labels;
  cv::Mat normal_centers;
  for(size_t i= 0; i < filtered_normals->size(); i++)
  {
    cv_filtered_normals.at<float>(i, 0)=
        filtered_normals->at(i).normal_x;
    cv_filtered_normals.at<float>(i, 1)=
        filtered_normals->at(i).normal_y;
    cv_filtered_normals.at<float>(i, 2)=
        filtered_normals->at(i).normal_z;
  }

  cv::kmeans(
      cv_filtered_normals, cluster_count, labels,
      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
      5, cv::KMEANS_PP_CENTERS, normal_centers);
  std::cout << normal_centers << endl;

  /*4 clusters maybe to large, merge similar normal and cloud*/
  ROS_INFO_STREAM("merge similar normals");
  vector<vector<int>> distinct_normal_indices;
  vector<int> first_indice;
  first_indice.push_back(0);
  distinct_normal_indices.push_back(first_indice);
  for(size_t i= 1; i < cluster_count; i++)
  {
    bool is_distinct= true;
    for(size_t j= 0; j < distinct_normal_indices.size(); j++)
    {
      int id= distinct_normal_indices.at(j).at(0);
      ROS_INFO_STREAM("id is" << id);
      double dot_product= normal_centers.at<float>(id, 0) *
                              normal_centers.at<float>(i, 0) +
                          normal_centers.at<float>(id, 1) *
                              normal_centers.at<float>(i, 1) +
                          normal_centers.at<float>(id, 2) *
                              normal_centers.at<float>(i, 2);
      if(dot_product > 0.7)
      {
        // a similar normal
        distinct_normal_indices.at(j).push_back(i);
        is_distinct= false;
        break;
      }
      else
      {
        // a distintc normal,continue to compare with other
      }
    }
    if(is_distinct)
    {
      vector<int> distinct_normal_indice;
      distinct_normal_indice.push_back(i);
      distinct_normal_indices.push_back(distinct_normal_indice);
    }
  }

  /*seperate different groups of normals*/
  for(size_t i= 0; i < distinct_normal_indices.size(); i++)
  {
    pcl::PointCloud<pcl::Normal>::Ptr normals_ptr(
        new pcl::PointCloud<pcl::Normal>);
    final_cluster_normals.push_back(normals_ptr);
    pcl::PointCloud<PointT>::Ptr cloud_ptr(
        new pcl::PointCloud<PointT>);
    final_cluster_cloud.push_back(cloud_ptr);
  }
  for(size_t i= 0; i < filtered_normals->size(); i++)
  {
    int cluster_id= -1;
    for(size_t j= 0; j < distinct_normal_indices.size(); j++)
    {
      bool is_member= false;
      for(size_t k= 0; k < distinct_normal_indices.at(j).size(); k++)
      {
        if(labels.at<int>(i, 0) ==
           distinct_normal_indices.at(j).at(k))
        {
          is_member= true;
        }
      }
      if(is_member)
      {
        cluster_id= j;
        break;
      }
    }
    final_cluster_normals.at(cluster_id)
        ->push_back(filtered_normals->at(i));
    final_cluster_cloud.at(cluster_id)
        ->push_back(filtered_cloud->at(i));
  }

  /*print cluster result*/
  for(size_t i= 0; i < distinct_normal_indices.size(); i++)
  {
    ROS_INFO_STREAM("cluster: " << i);
    for(size_t j= 0; j < distinct_normal_indices.at(i).size(); j++)
    {
      ROS_INFO_STREAM(distinct_normal_indices.at(i).at(j));
    }
  }
}

/**visualize cluster normal and cloud*/
void DroneExplorer::visualizeNormalsAndCloud(
    vector<pcl::PointCloud<pcl::Normal>::Ptr> final_cluster_normals,
    vector<pcl::PointCloud<PointT>::Ptr> final_cluster_cloud)
{
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  viewer.setBackgroundColor(0.0, 0.0, 0.0);
  int rgb_array[4][3]= {
    { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 0 }
  };
  /*add different cluster of normals and cloud*/
  for(size_t i= 0; i < final_cluster_normals.size(); i++)
  {
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
        rgb_handler(final_cluster_cloud.at(i), rgb_array[i][0],
                    rgb_array[i][1], rgb_array[i][2]);
    viewer.addPointCloud<pcl::PointXYZRGB>(final_cluster_cloud.at(i),
                                           rgb_handler, "cloud" + i);
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
        "cloud" + i);
    viewer.addPointCloudNormals<PointT, pcl::Normal>(
        final_cluster_cloud.at(i), final_cluster_normals.at(i), 10,
        0.2, "normal" + i);
  }
  viewer.addCoordinateSystem(1.0);

  while(!viewer.wasStopped())
  {
    viewer.spinOnce();
  }
}
