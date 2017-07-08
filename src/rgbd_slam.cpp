// the following are UBUNTU/LINUX ONLY terminal color
#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

#include "rgb_d_slam.h"
#include "parameter_reader.h"
using namespace rgbd;

// note://///////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
/**image transport used to receive rgbd camera image*/
image_transport::Subscriber rgb_image_sub;
image_transport::Subscriber depth_image_sub;
cv::Mat g_rgb_image;
cv::Mat g_depth_image;
/***************************************************************

    variable  definition

***********************************************************/
int taskMode= 0;  ///<  0 : visual odometry, 1 use octomap

// 把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>
    SlamLinearSolver;
/*******************************************************

    function  definition

******************************************************/

// rgb and depth image callback
void rgbImageCallback(const sensor_msgs::Image::ConstPtr msg);
void depthImageCallback(const sensor_msgs::Image::ConstPtr msg);
// print octree infomation
void print_query_info(point3d query, ColorOcTreeNode *node);

int main(int argc, char **argv)
{
  /*******************************************************
*
*                                                     ROS
*initialization
*
************************************************************/
  ros::init(argc, argv, "rgbd_slam");
  ros::NodeHandle node;
  ros::Rate loop_rate(50);

  /*subscribe to rgb and depth image of rgbd camera*/
  image_transport::ImageTransport image_transport(node);
  rgb_image_sub= image_transport.subscribe("/camera/rgb/image_raw", 1,
                                           rgbImageCallback);
  depth_image_sub= image_transport.subscribe("/camera/depth/image", 1,
                                             depthImageCallback);

  /*read parameter*/
  ParameterReader pd;
  taskMode= atoi(pd.getData("task_mode").c_str());
  if(taskMode == 0)
  {
    /*visual odometry and mapping , loop enclosure and graph
     * optimization*/
    int startIndex= atoi(pd.getData("start_index").c_str());
    int endIndex= atoi(pd.getData("end_index").c_str());
    vector<FRAME> keyframes; /*keyframe*/
    vector<FRAME> frames_to_be_map;
    /*initialize the first frame*/
    cout << "Initializing ..." << endl;
    int currIndex= startIndex;  // 当前索引为currIndex
    FRAME currFrame= readFrame(currIndex, pd);  // 上一帧数据
    currFrame.cameraPos[0]= currFrame.cameraPos[1]=
        currFrame.cameraPos[2]= 0;

    string detector= pd.getData("detector");
    string descriptor= pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera= getDefaultCamera();
    cout << "Ifirst compute key point ..." << endl;
    computeKeyPointsAndDesp(currFrame, detector, descriptor);
    cout << "compute keypoint finish.." << endl;
    PointCloud::Ptr cloud=
        image2PointCloud(currFrame.rgb, currFrame.depth, camera);

    // 有关g2o的初始化
    /* 初始化求解器*/
    SlamLinearSolver *linearSolver= new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver *blockSolver= new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver=
        new g2o::OptimizationAlgorithmLevenberg(blockSolver);
    g2o::SparseOptimizer globalOptimizer;  // 最后用的就是这个东东
    globalOptimizer.setAlgorithm(solver);
    globalOptimizer.setVerbose(false);  // 不要输出调试信息

    /*向globalOptimizer增加第一个顶点*/
    g2o::VertexSE3 *v= new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity());  //估计为单位矩阵
    v->setFixed(true);  //第一个顶点固定，不用优化
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);
    frames_to_be_map.push_back(currFrame);

    /*initialze slam parameters*/
    double keyframe_threshold=
        atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure=
        pd.getData("check_loop_closure") == string("yes");
    const int optimization_frames_number=
        10;  // optimize every 10 frames

    /*initialize point cloud filters*/
    /*网格滤波器，调整地图分辨率*/
    pcl::VoxelGrid<PointT> voxel;
    double gridsize= atof(pd.getData("voxel_grid").c_str());
    voxel.setLeafSize(gridsize, gridsize, gridsize);
    /*由于rgbd相机的有效深度区间有限，把太远的去掉,4m以上就不要了*/
    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0);
    /* outliers removal*/
    pcl::StatisticalOutlierRemoval<PointT> outlierRemoval;
    int meanK= atoi(pd.getData("mean_k").c_str());
    float stdDev= atof(pd.getData("std_dev").c_str());
    outlierRemoval.setMeanK(meanK);
    outlierRemoval.setStddevMulThresh(stdDev);

    /*initialze global point cloud map and octomap*/
    PointCloud::Ptr globalOutput(new PointCloud());  //全局地图
    double treeResolution=
        atof(pd.getData("tree_resolution").c_str());
    octomap::ColorOcTree tree(treeResolution);  // global octomap

    /*localizing and mapping for dataset*/
    for(currIndex= startIndex + 1; currIndex < endIndex && ros::ok();
        currIndex++)
    {
      cout << "Reading files " << currIndex << endl;
      FRAME currFrame= readFrame(currIndex, pd);
      computeKeyPointsAndDesp(currFrame, detector, descriptor);
      //匹配该帧与keyframes里最后一帧
      CHECK_RESULT result= checkKeyframes(keyframes.back(), currFrame,
                                          globalOptimizer);
      // 根据匹配结果不同采取不同策略
      switch(result)
      {
        case NOT_MATCHED:
          //没匹配上，直接跳过
          cout << "Not enough inliers." << endl;
          break;
        case TOO_FAR_AWAY:
          // 太近了，也直接跳
          cout << "Too far away, may be an error." << endl;
          break;
        case TOO_CLOSE:
          // 太远了，可能出错了
          cout << "Too close, not a keyframe" << endl;
          break;
        case KEYFRAME:
          cout << "This is a new keyframe" << endl;
          // 检测回环
          if(check_loop_closure)
          {
            checkNearbyLoops(keyframes, currFrame, globalOptimizer);
            checkRandomLoops(keyframes, currFrame, globalOptimizer);
          }
          keyframes.push_back(currFrame);
          frames_to_be_map.push_back(currFrame);

          /*optimize when have enough new frame*/
          if(frames_to_be_map.size() > optimization_frames_number)
          {
            /*  add new points to map */
            cout << "Now add new points to map" << endl;
            /*optimization*/
            cout << "optimizing pose graph, vertices: "
                 << globalOptimizer.vertices().size() << endl;
            // globalOptimizer.save(
            // "/home/zby/uav_slam_ws/src/rgbd_slam/result_before.g2o");
            globalOptimizer.initializeOptimization();
            globalOptimizer.optimize(100);
            // globalOptimizer.save(
            // "~/uav_slam_ws/src/rgbd_slam/result_after.g2o" );
            cout << "Optimization done." << endl;

            /*拼接点云地图*/
            PointCloud::Ptr localOutput(new PointCloud());  //局部地图
            // point cloud transformed
            PointCloud::Ptr pct(new PointCloud());
            // point cloud filtered
            PointCloud::Ptr pcf(new PointCloud());
            for(size_t i= 0; i < frames_to_be_map.size(); i++)
            {
              g2o::VertexSE3 *vertex= dynamic_cast<g2o::VertexSE3 *>(
                  globalOptimizer.vertex(
                      frames_to_be_map[i].frameID));
              /*该帧优化后的位姿*/
              Eigen::Isometry3d pose= vertex->estimate();
              PointCloud::Ptr newCloud=
                  image2PointCloud(frames_to_be_map[i].rgb,
                                   frames_to_be_map[i].depth, camera);
              /*网格滤波，降采样*/
              voxel.setInputCloud(newCloud);
              voxel.filter(*pcf);
              pass.setInputCloud(pcf);
              pass.filter(*newCloud);
              /*把点云变换后加入全局地图中*/
              pcl::transformPointCloud(*newCloud, *pct,
                                       pose.matrix());
              *localOutput+= *pct;
              pct->clear();
              newCloud->clear();
              /*filter point cloud*/
              voxel.setInputCloud(localOutput);
              voxel.filter(*pcf);
              outlierRemoval.setInputCloud(pcf);
              outlierRemoval.filter(*localOutput);
              /*add to global point cloud*/
              *globalOutput+= *localOutput;
              cout << "point cloud update done" << endl;

              /*generate octomap*/
              for(auto p : localOutput->points)
              {
                // use  insertRay to update free points
                point3d cp(frames_to_be_map.at(i).cameraPos[0],
                           frames_to_be_map.at(i).cameraPos[1],
                           frames_to_be_map.at(i).cameraPos[2]);
                tree.insertRay(cp, octomap::point3d(p.x, p.y, p.z));
              }
              for(auto p : localOutput->points)
              {
                // update occupied points
                ColorOcTreeNode *node=
                    tree.search(point3d(p.x, p.y, p.z));
                node->setLogOdds(2.2);
                // cout<<"current logOdds is:
                // "<<node->getLogOdds()<<endl
                //         <<" occupancy is
                //         :"<<node->getOccupancy()<<endl;
                // tree.updateNode(octomap::point3d(p.x,p.y,p.z),true);
              }
              for(auto p : localOutput->points)
              {
                tree.integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
              }
              tree.updateInnerOccupancy();
              cout << "octomap update done." << endl;
              localOutput->clear();
            }
            frames_to_be_map.clear();
          }
          break;
        default:
          break;
      }
    }
    //存储
    pcl::io::savePCDFile("/home/zby/uav_slam_ws/src/rgbd_slam/"
                         "generated/rgbdslam_result.pcd",
                         *globalOutput);
    tree.write("/home/zby/uav_slam_ws/src/rgbd_slam/generated/"
               "octo_free.ot");
    cout << "Final map is saved." << endl;
  }

  else if(taskMode == 1)
  {
    /*****************************************************
*
*                                                    use octmap to
*navigate
*
********************************************************/
    // read a .ot file
    AbstractOcTree *tree= AbstractOcTree::read(
        "/home/zby/uav_slam_ws/src/rgbd_slam/generated/octo_free.ot");
    ColorOcTree *ct= dynamic_cast<ColorOcTree *>(tree);
    double treeResolution=
        atof(pd.getData("tree_resolution").c_str());
    cout << "Read ot file finish" << endl;
    /*********************************************************
search a built tree‘s leaves to find the frontiers voxel grid
*********************************************************/
    int totalCount= 0;
    point3d_collection frontiers;
    for(ColorOcTree::leaf_iterator it= ct->begin_leafs(),
                                   end= ct->end_leafs();
        it != end; ++it)
    {
      totalCount++;
      // get some parameters of node
      point3d node= it.getCoordinate();
      double occupancy= it->getOccupancy();
      if(occupancy > 0.75)
        continue;
      // std::cout << "Free VoxelGrid center: " <<node<< std::endl;
      // get nearest 26 points' coordinates
      for(int i= -1; i <= 1; i++)
      {
        for(int j= -1; j <= 1; j++)
        {
          for(int k= -1; k <= 1; k++)
          {
            if(i == 0 && j == 0 && k == 0)
              continue;
            point3d pt(node.x() + i * treeResolution,
                       node.y() + j * treeResolution,
                       node.z() + k * treeResolution);
            ColorOcTreeNode *nearNode= ct->search(pt);
            // print_query_info(pt,ctn);
            if(nearNode != NULL)
            {
              // cout << "occupancy probability at " << query << ":\t
              // " <<
              // node->getOccupancy() << endl;
              // cout<<"not null"<<endl;
            }
            else
            {
              // cout << "occupancy probability at " << pt << ":\t is
              // unknown"
              // << endl;
              // cout<<"null node !"<<endl;
              // add a frontier node
              frontiers.push_back(pt);
              // i=j=k=2;
              // cout<<i<<j<<k<<endl;
            }
          }
        }
      }
    }
    cout << "leaf count is :" << totalCount << endl;
    cout << "frontier number is:" << frontiers.size() << endl;
    /*********************************************************
visulize:add frontier voxel grid to ocTree,
*********************************************************/
    for(int i= 0; i < frontiers.size(); i++)
    {
      point3d pt= frontiers.at(i);
      ct->updateNode(pt, true);
      ct->integrateNodeColor(pt.x(), pt.y(), pt.z(), 255, 255, 255);
      ct->setNodeColor(pt.x(), pt.y(), pt.z(), 255, 255, 255);
    }
    // ct->updateInnerOccupancy();
    /*********************************************************
visulize:add origin and x,y,z axials
*********************************************************/
    ct->updateNode(point3d(0, 0, 0), true);
    ct->integrateNodeColor(0, 0, 0, 255, 255, 255);
    for(int i= 1; i <= 20; i++)
    {
      ct->updateNode(point3d(treeResolution * i, 0, 0), true);
      ct->integrateNodeColor(treeResolution * i, 0, 0, 255, 0, 0);
      ct->updateNode(point3d(0, treeResolution * i, 0), true);
      ct->integrateNodeColor(0, treeResolution * i, 0, 0, 255, 0);
      ct->updateNode(point3d(0, 0, treeResolution * i), true);
      ct->integrateNodeColor(0, 0, treeResolution * i, 0, 0, 255);
    }
    ct->updateInnerOccupancy();
    ct->write("/home/zby/uav_slam_ws/src/rgbd_slam/generated/"
              "octo_frontier.ot");
    /*********************************************************
test how to change a node's value
*********************************************************/
    // //initial
    // point3d testPoint(6,6,6);
    // ct->updateNode(testPoint,true);
    // ColorOcTreeNode *node=ct->search(testPoint);
    // print_query_info(testPoint,node);
    // //set false by update
    // ct->updateNode(testPoint,false);
    // print_query_info(testPoint,node);
    // //set true by update
    // ct->updateNode(testPoint,true);
    // print_query_info(testPoint,node);
    // //set false by insert ray
    // ct->insertRay(octomap::point3d(5,5,5),octomap::point3d(7,7,7));
    // print_query_info(testPoint,node);
    // //set true by insert ray
    // ct->insertRay(octomap::point3d(5,5,5),octomap::point3d(6,6,6));
    // print_query_info(testPoint,node);

    /*************************************************
get frontiers points' coordinate and calculate normals
**************************************************/
    /*create point cloud from octomap frontier*/
    PointCloud::Ptr frontierCloud(new PointCloud);
    for(int i= 0; i < frontiers.size(); i++)
    {
      point3d pt= frontiers.at(i);
      PointT p;
      p.x= pt.x();
      p.y= pt.y();
      p.z= pt.z();
      p.r= p.g= p.g= 255;
      frontierCloud->points.push_back(p);
    }
    frontierCloud->height= 1;
    frontierCloud->width= frontierCloud->points.size();
    frontierCloud->is_dense= false;

    /*estimate normals using pcl */
    pcl::NormalEstimation<PointT, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(frontierCloud);
    pcl::search::KdTree<PointT>::Ptr kdTree(
        new pcl::search::KdTree<PointT>());
    normalEstimator.setSearchMethod(kdTree);
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(
        new pcl::PointCloud<pcl::Normal>);
    /*Use all neighbors in a sphere of radius 50cm*/
    double neighborRadius=
        atof(pd.getData("neighbor_radius").c_str());
    normalEstimator.setRadiusSearch(neighborRadius);
    normalEstimator.compute(*cloudNormals);
    // /*output normals' value*/
    // for (size_t i = 0; i < cloudNormals->size(); i++) {
    //   ROS_INFO_STREAM("normal  " << i << "is: " <<
    //   cloudNormals->at(i).normal_x
    //                              << "," <<
    //                              cloudNormals->at(i).normal_y <<
    //                              ","
    //                              << cloudNormals->at(i).normal_z);
    // }

    /*filter normals*/
    double unit_normal_z[3]= { 0, 1.0, 0 };
    vector<int> normal_indices;
    for(size_t i= 0; i < cloudNormals->size(); i++)
    {
      double dot_product=
          unit_normal_z[0] * cloudNormals->at(i).normal_x +
          unit_normal_z[1] * cloudNormals->at(i).normal_y +
          unit_normal_z[2] * cloudNormals->at(i).normal_z;
      // ROS_INFO_STREAM("dot_product " << i << "is :" <<
      // dot_product);
      if(fabs(dot_product) < 0.5)
        normal_indices.push_back(i);
    }
    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals(
        new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr filtered_cloud(
        new pcl::PointCloud<PointT>);
    for(size_t i= 0; i < normal_indices.size(); i++)
    {
      filtered_normals->push_back(
          cloudNormals->at(normal_indices.at(i)));
      filtered_cloud->push_back(
          frontierCloud->at(normal_indices.at(i)));
    }
    cout << "frontier points' size: " << frontierCloud->points.size()
         << endl;
    ROS_INFO_STREAM(
        "filtered_normals size is :" << filtered_normals->size());
    ROS_INFO_STREAM(
        "filtered_cloud size is :" << filtered_cloud->size());

    /**use kmeans cluster to classify different surface*/
    int cluster_count= 4;
    cv::Mat cv_filtered_normals(filtered_normals->size(), 3, CV_32F);
    // cout << cv_filtered_normals << endl;
    cv::Mat labels;
    cv::Mat normal_centers;
    for(size_t i= 0; i < filtered_normals->size(); i++)
    {
      ROS_INFO_STREAM("converting normal" << i);
      cv_filtered_normals.at<float>(i, 0)=
          filtered_normals->at(i).normal_x;
      cv_filtered_normals.at<float>(i, 1)=
          filtered_normals->at(i).normal_y;
      cv_filtered_normals.at<float>(i, 2)=
          filtered_normals->at(i).normal_z;
      ROS_INFO_STREAM("normal "
                      << i << "is "
                      << cv_filtered_normals.at<float>(i, 0) << ","
                      << cv_filtered_normals.at<float>(i, 1) << ","
                      << cv_filtered_normals.at<float>(i, 2));
    }
    // cout << cv_filtered_normals << endl;
    ROS_INFO_STREAM("begin kmeans");
    // ros::Duration(30.0).sleep();

    cv::kmeans(
        cv_filtered_normals, cluster_count, labels,
        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
        5, cv::KMEANS_PP_CENTERS, normal_centers);
    /**merge similar normal and cloud*/
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
    for(size_t` i= 0; i < distinct_normal_indices.size(); i++)
    {
      ROS_INFO_STREAM("cluster: " << i);
      for(size_t j= 0; j < distinct_normal_indices.at(i).size(); j++)
      {
        ROS_INFO_STREAM(distinct_normal_indices.at(i).at(j));
      }
    }
    // int count_labels[4] = {0, 0, 0, 0};
    // for (int i = 0; i < filtered_normals->size(); i++) {
    //   if (labels.at<int>(i, 0) == 0)
    //     count_labels[0]++;
    //   else if (labels.at<int>(i, 0) == 1)
    //     count_labels[1]++;
    //   else if (labels.at<int>(i, 0) == 2)
    //     count_labels[2]++;
    //   else if (labels.at<int>(i, 0) == 3)
    //     count_labels[3]++;
    // }
    // ROS_INFO_STREAM("label count is:"
    //                 << count_labels[0] << "," << count_labels[1] <<
    //                 ","
    //                 << count_labels[2] << "," << count_labels[3]);
    // for (size_t i = 0; i < filtered_normals->size(); i++) {
    //   // ROS_INFO_STREAM("normal " << i << "'s label is:" <<
    //   labels.at<int>(i));
    // }
    /*seperate different groups of normals*/
    vector<pcl::PointCloud<pcl::Normal>::Ptr> final_cluster_normals;
    vector<pcl::PointCloud<PointT>::Ptr> final_cluster_cloud;
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
        for(size_t k= 0; k < distinct_normal_indices.at(j).size();
            k++)
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
    for(size_t i= 0; i < final_cluster_normals.size(); i++)
    {
      ROS_INFO_STREAM("normal cluster " << i << " has "
                                        << final_cluster_normals.at(i)
                                               ->size() << "normals");
      ROS_INFO_STREAM("cloud cluster " << i << " has "
                                       << final_cluster_cloud.at(i)
                                              ->size() << "points");
    }

    /**visualize cluster normal and cloud*/
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    int rgb_array[4][3]= {
      { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 0 }
    };
    /*add different cluster of normals and cloud*/
    for(size_t i= 0; i < final_cluster_normals.size(); i++)
    {
      pcl::visualization::PointCloudColorHandlerCustom<
          pcl::PointXYZRGB> rgb_handler(final_cluster_cloud.at(i),
                                        rgb_array[i][0],
                                        rgb_array[i][1],
                                        rgb_array[i][2]);
      viewer.addPointCloud<pcl::PointXYZRGB>(
          final_cluster_cloud.at(i), rgb_handler, "cloud" + i);
      viewer.setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
          "cloud" + i);
      viewer.addPointCloudNormals<PointT, pcl::Normal>(
          final_cluster_cloud.at(i), final_cluster_normals.at(i), 10,
          0.2, "normal" + i);
    }
    viewer.addCoordinateSystem(1.0);
    /**visualize origin normal and cloud*/
    pcl::visualization::PCLVisualizer viewer_o("PCL Viewer Origin");
    viewer_o.setBackgroundColor(0.0, 0.0, 0.0);
    /*add origin normals and cloud*/
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
        rgb_handler_o(frontierCloud, 255, 255, 0);
    viewer_o.addPointCloud<pcl::PointXYZRGB>(frontierCloud,
                                             rgb_handler_o, "cloud");
    viewer_o.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    // viewer_o.addPointCloudNormals<PointT, pcl::Normal>(
    //     filtered_cloud, filtered_normals, 10, 0.2, "normal");
    viewer_o.addCoordinateSystem(1.0);
    while(!viewer.wasStopped())
    {
      viewer.spinOnce();
    }
  }

  return 0;
}

/*subscribe to rgb image*/
void rgbImageCallback(const sensor_msgs::Image::ConstPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr=
        cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch(cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  g_rgb_image= cv_ptr->image;
}

/*subscribe to depth image*/
void depthImageCallback(const sensor_msgs::Image::ConstPtr msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr= cv_bridge::toCvCopy(
        msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch(cv_bridge::Exception &e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  g_depth_image= cv_ptr->image;
}

void print_query_info(point3d query, ColorOcTreeNode *node)
{
  if(node != NULL)
  {
    std::cout << query << "\t"
              << "Occupancy is: " << node->getOccupancy() << endl
              << " logOdds is :" << node->getLogOdds() << endl
              << " Value is: " << node->getValue() << endl;
  }
  else
    cout << "occupancy probability at " << query << ":\t is unknown"
         << endl;
}
