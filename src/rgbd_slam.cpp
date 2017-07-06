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
// C++标准库
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
using namespace std;
// g2o
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/types_slam3d.h>
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
// OpenCV
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/xfeatures2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>
// PCL
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
// octomap
#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>
using namespace octomap;
// ros
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int16.h>
// note://///////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
/***************************************************************

    variable  definition

***********************************************************/
int taskMode = 0; ///<  0 : visual odometry, 1 use octomap
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
class ParameterReader {
public:
  ParameterReader(
      string filename = "/home/zby/uav_slam_ws/src/rgbd_slam/parameters.txt") {
    ifstream fin(filename.c_str());
    if (!fin) {
      cerr << "parameter file does not exist." << endl;
      return;
    }
    while (!fin.eof()) {
      string str;
      getline(fin, str);
      if (str[0] == '#') {
        // 以‘＃’开头的是注释
        continue;
      }

      int pos = str.find("=");
      if (pos == -1)
        continue;
      string key = str.substr(0, pos);
      string value = str.substr(pos + 1, str.length());
      data[key] = value;

      if (!fin.good())
        break;
    }
  }
  string getData(string key) {
    map<string, string>::iterator iter = data.find(key);
    if (iter == data.end()) {
      cerr << "Parameter name " << key << " not found!" << endl;
      return string("NOT_FOUND");
    }
    return iter->second;
  }

public:
  map<string, string> data;
};
/**相机内参结构*/
struct CAMERA_INTRINSIC_PARAMETERS {
  double cx, cy, fx, fy, scale;
};
// 帧结构
struct FRAME {
  int frameID;
  cv::Mat rgb, depth;      //该帧对应的彩色图与深度图
  cv::Mat desp;            //特征描述子
  vector<cv::KeyPoint> kp; //关键点
  double cameraPos[3];
};
// PnP 结果
struct RESULT_OF_PNP {
  cv::Mat rvec, tvec;
  int inliers;
};
// 把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>
    SlamLinearSolver;
/*******************************************************

    function  definition

******************************************************/
// 给定index，读取一帧数据
FRAME
readFrame(int index, ParameterReader &pd);
/**估计一个运动的大小
* @param rvec rotation vector
* @param tvec translation vector
* @return a value represent how far the
*   motion is
*/
double normofTransform(cv::Mat rvec, cv::Mat tvec);
// 检测两个帧，结果定义
enum CHECK_RESULT { NOT_MATCHED = 0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME };
//检查是否为关键帧
CHECK_RESULT
checkKeyframes(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti,
               bool is_loops = false);
// 检测近距离的回环
void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame,
                      g2o::SparseOptimizer &opti);
// 随机检测回环
void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame,
                      g2o::SparseOptimizer &opti);
// 函数接口
// image2PonitCloud 将rgb图转换为点云
PointCloud::Ptr image2PointCloud(cv::Mat &rgb, cv::Mat &depth,
                                 CAMERA_INTRINSIC_PARAMETERS &camera);
// point2dTo3d 将单个点从图像坐标转换为空间坐标
// input: 3维点Point3f (u,v,d)
cv::Point3f point2dTo3d(cv::Point3f &point,
                        CAMERA_INTRINSIC_PARAMETERS &camera);
// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp(FRAME &frame, string detector, string descriptor);
// estimateMotion 计算两个帧之间的运动
// 输入：帧1和帧2, 相机内参
RESULT_OF_PNP
estimateMotion(FRAME &frame1, FRAME &frame2,
               CAMERA_INTRINSIC_PARAMETERS &camera);
// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen(cv::Mat &rvec, cv::Mat &tvec);
// joinPointCloud
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME &newFrame,
                               Eigen::Isometry3d T,
                               CAMERA_INTRINSIC_PARAMETERS &camera);
// get camera intristic parameter
CAMERA_INTRINSIC_PARAMETERS
getDefaultCamera();
// rgb and depth image callback
void rgbImageCallback(const sensor_msgs::Image::ConstPtr msg);
void depthImageCallback(const sensor_msgs::Image::ConstPtr msg);
// print octree infomation
void print_query_info(point3d query, ColorOcTreeNode *node);

int main(int argc, char **argv) {
  /*******************************************************
*
*                                                     ROS initialization
*
************************************************************/
  ros::init(argc, argv, "rgbd_slam");
  ros::NodeHandle node;
  ros::Rate loop_rate(50);

  // ros::Publisher pub_twist;
  // ros::Subscriber sub_rgb_image;
  image_transport::ImageTransport it_rgb_image(node);
  image_transport::ImageTransport it_depth_image(node);
  // /camera/rgb/image_rect_color
  //             /camera/depth/image
  //             sensor_msgs/Image
  image_transport::Subscriber sub_rgb_image = it_rgb_image.subscribe(
      "/camera/rgb/image_rect_color", 1, rgbImageCallback);
  image_transport::Subscriber sub_depth_image =
      it_depth_image.subscribe("/camera/depth/image", 1, depthImageCallback);

  ParameterReader pd;
  taskMode = atoi(pd.getData("task_mode").c_str());
  if (taskMode == 0) {
    /******************************************************
*
*                                                           visual odometry
*and mapping
*
*********************************************************/
    int startIndex = atoi(pd.getData("start_index").c_str());
    int endIndex = atoi(pd.getData("end_index").c_str());
    vector<FRAME> keyframes; /*keyframe*/
    vector<FRAME> frameToMap;
    /*initialize the first frame*/
    cout << "Initializing ..." << endl;
    int currIndex = startIndex;                 // 当前索引为currIndex
    FRAME currFrame = readFrame(currIndex, pd); // 上一帧数据
    currFrame.cameraPos[0] = currFrame.cameraPos[1] = currFrame.cameraPos[2] =
        0;

    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    cout << "Ifirst compute key point ..." << endl;
    computeKeyPointsAndDesp(currFrame, detector, descriptor);
    cout << "compute keypoint finish.." << endl;
    PointCloud::Ptr cloud =
        image2PointCloud(currFrame.rgb, currFrame.depth, camera);

    // 有关g2o的初始化
    // 初始化求解器
    SlamLinearSolver *linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering(false);
    SlamBlockSolver *blockSolver = new SlamBlockSolver(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver =
        new g2o::OptimizationAlgorithmLevenberg(blockSolver);

    g2o::SparseOptimizer globalOptimizer; // 最后用的就是这个东东
    globalOptimizer.setAlgorithm(solver);
    globalOptimizer.setVerbose(false); // 不要输出调试信息

    // 向globalOptimizer增加第一个顶点
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(currIndex);
    v->setEstimate(Eigen::Isometry3d::Identity()); //估计为单位矩阵
    v->setFixed(true); //第一个顶点固定，不用优化
    globalOptimizer.addVertex(v);

    keyframes.push_back(currFrame);
    frameToMap.push_back(currFrame);

    double keyframe_threshold = atof(pd.getData("keyframe_threshold").c_str());
    bool check_loop_closure = pd.getData("check_loop_closure") == string("yes");

    const int optiThre = 10;      // optimize every 10 frames
    pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
    double gridsize =
        atof(pd.getData("voxel_grid").c_str()); //分辨图可以在parameters.txt里调
    voxel.setLeafSize(gridsize, gridsize, gridsize);
    pcl::PassThrough<PointT>
        pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 4.0); // 4m以上就不要了
    pcl::StatisticalOutlierRemoval<PointT> outlierRemoval; // outliers removal
    int meanK = atoi(pd.getData("mean_k").c_str());
    float stdDev = atof(pd.getData("std_dev").c_str());
    outlierRemoval.setMeanK(meanK);
    outlierRemoval.setStddevMulThresh(stdDev);
    PointCloud::Ptr globalOutput(new PointCloud()); //全局地图
    double treeResolution = atof(pd.getData("tree_resolution").c_str());
    octomap::ColorOcTree tree(treeResolution); // global octomap
    for (currIndex = startIndex + 1; currIndex < endIndex && ros::ok();
         currIndex++) {
      cout << "Reading files " << currIndex << endl;
      FRAME currFrame = readFrame(currIndex, pd); // 读取currFrame
      computeKeyPointsAndDesp(currFrame, detector, descriptor); //提取特征
      CHECK_RESULT result =
          checkKeyframes(keyframes.back(), currFrame,
                         globalOptimizer); //匹配该帧与keyframes里最后一帧
      switch (result) // 根据匹配结果不同采取不同策略
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
        if (check_loop_closure) {
          checkNearbyLoops(keyframes, currFrame, globalOptimizer);
          checkRandomLoops(keyframes, currFrame, globalOptimizer);
        }
        keyframes.push_back(currFrame);
        frameToMap.push_back(currFrame);
        // optimize when enough frame accumulate
        if (frameToMap.size() > optiThre) {
          /*  add new points to map */
          cout << "Now add new points to map" << endl;
          /*optimization*/
          cout << "optimizing pose graph, vertices: "
               << globalOptimizer.vertices().size() << endl;
          // globalOptimizer.save(
          // "/home/zby/uav_slam_ws/src/rgbd_slam/result_before.g2o");
          globalOptimizer.initializeOptimization();
          globalOptimizer.optimize(100); //可以指定优化步数
          // globalOptimizer.save(
          // "~/uav_slam_ws/src/rgbd_slam/result_after.g2o" );
          cout << "Optimization done." << endl;
          /*拼接点云地图*/
          PointCloud::Ptr localOutput(new PointCloud()); //局部地图
          PointCloud::Ptr pct(new PointCloud()); // point cloud transformed
          PointCloud::Ptr pcf(new PointCloud()); // point cloud filtered
          for (size_t i = 0; i < frameToMap.size(); i++) {
            /*从g2o里取出一帧*/
            g2o::VertexSE3 *vertex = dynamic_cast<g2o::VertexSE3 *>(
                globalOptimizer.vertex(frameToMap[i].frameID));
            Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
            PointCloud::Ptr newCloud = image2PointCloud(
                frameToMap[i].rgb, frameToMap[i].depth, camera); //转成点云
            /*网格滤波，降采样*/
            voxel.setInputCloud(newCloud);
            voxel.filter(*pcf);
            pass.setInputCloud(pcf);
            pass.filter(*newCloud);
            /*把点云变换后加入全局地图中*/
            pcl::transformPointCloud(*newCloud, *pct, pose.matrix());
            *localOutput += *pct;
            pct->clear();
            newCloud->clear();
            /*filter point cloud*/
            voxel.setInputCloud(localOutput);
            voxel.filter(*pcf);
            outlierRemoval.setInputCloud(pcf);
            outlierRemoval.filter(*localOutput);
            /*add to global point cloud*/
            *globalOutput += *localOutput;
            cout << "point cloud update done" << endl;
            /*generate octomap*/
            for (auto p : localOutput->points) {
              // use  insertRay to update free points
              point3d cp(frameToMap.at(i).cameraPos[0],
                         frameToMap.at(i).cameraPos[1],
                         frameToMap.at(i).cameraPos[2]);
              tree.insertRay(cp, octomap::point3d(p.x, p.y, p.z));
            }
            for (auto p : localOutput->points) {
              // update occupied points
              ColorOcTreeNode *node = tree.search(point3d(p.x, p.y, p.z));
              node->setLogOdds(2.2);
              // cout<<"current logOdds is: "<<node->getLogOdds()<<endl
              //         <<" occupancy is :"<<node->getOccupancy()<<endl;
              // tree.updateNode(octomap::point3d(p.x,p.y,p.z),true);
            }
            for (auto p : localOutput->points) {
              tree.integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);
            }
            tree.updateInnerOccupancy();
            cout << "octomap update done." << endl;
            localOutput->clear();
          }
          frameToMap.clear();
        }
        break;
      default:
        break;
      }
    }
    //存储
    pcl::io::savePCDFile(
        "/home/zby/uav_slam_ws/src/rgbd_slam/generated/rgbdslam_result.pcd",
        *globalOutput);
    tree.write("/home/zby/uav_slam_ws/src/rgbd_slam/generated/octo_free.ot");
    cout << "Final map is saved." << endl;
  }

  else if (taskMode == 1) {
    /*****************************************************
*
*                                                    use octmap to navigate
*
********************************************************/
    // read a .ot file
    AbstractOcTree *tree = AbstractOcTree::read(
        "/home/zby/uav_slam_ws/src/rgbd_slam/generated/octo_free.ot");
    ColorOcTree *ct = dynamic_cast<ColorOcTree *>(tree);
    double treeResolution = atof(pd.getData("tree_resolution").c_str());
    cout << "Read ot file finish" << endl;
    /*********************************************************
search a built tree‘s leaves to find the frontiers voxel grid
*********************************************************/
    int totalCount = 0;
    point3d_collection frontiers;
    for (ColorOcTree::leaf_iterator it = ct->begin_leafs(),
                                    end = ct->end_leafs();
         it != end; ++it) {
      totalCount++;
      // get some parameters of node
      point3d node = it.getCoordinate();
      double occupancy = it->getOccupancy();
      if (occupancy > 0.75)
        continue;
      // std::cout << "Free VoxelGrid center: " <<node<< std::endl;
      // get nearest 26 points' coordinates
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          for (int k = -1; k <= 1; k++) {
            if (i == 0 && j == 0 && k == 0)
              continue;
            point3d pt(node.x() + i * treeResolution,
                       node.y() + j * treeResolution,
                       node.z() + k * treeResolution);
            ColorOcTreeNode *nearNode = ct->search(pt);
            // print_query_info(pt,ctn);
            if (nearNode != NULL) {
              // cout << "occupancy probability at " << query << ":\t " <<
              // node->getOccupancy() << endl;
              // cout<<"not null"<<endl;
            } else {
              // cout << "occupancy probability at " << pt << ":\t is unknown"
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
    for (int i = 0; i < frontiers.size(); i++) {
      point3d pt = frontiers.at(i);
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
    for (int i = 1; i <= 20; i++) {
      ct->updateNode(point3d(treeResolution * i, 0, 0), true);
      ct->integrateNodeColor(treeResolution * i, 0, 0, 255, 0, 0);
      ct->updateNode(point3d(0, treeResolution * i, 0), true);
      ct->integrateNodeColor(0, treeResolution * i, 0, 0, 255, 0);
      ct->updateNode(point3d(0, 0, treeResolution * i), true);
      ct->integrateNodeColor(0, 0, treeResolution * i, 0, 0, 255);
    }
    ct->updateInnerOccupancy();
    ct->write("/home/zby/uav_slam_ws/src/rgbd_slam/generated/octo_frontier.ot");
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
    for (int i = 0; i < frontiers.size(); i++) {
      point3d pt = frontiers.at(i);
      PointT p;
      p.x = pt.x();
      p.y = pt.y();
      p.z = pt.z();
      p.r = p.g = p.g = 255;
      frontierCloud->points.push_back(p);
    }
    frontierCloud->height = 1;
    frontierCloud->width = frontierCloud->points.size();
    frontierCloud->is_dense = false;
    // pcl::io::savePCDFile(
    // "/home/zby/uav_slam_ws/src/rgbd_slam/generated/rgbdslam_frontier.pcd",
    // *frontierCloud);
    // cout<<" frontier cloud saving done"<<endl;

    /*estimate normals using pcl */
    pcl::NormalEstimation<PointT, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(frontierCloud);
    pcl::search::KdTree<PointT>::Ptr kdTree(new pcl::search::KdTree<PointT>());
    normalEstimator.setSearchMethod(kdTree);
    pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(
        new pcl::PointCloud<pcl::Normal>);
    /*Use all neighbors in a sphere of radius 50cm*/
    double neighborRadius = atof(pd.getData("neighbor_radius").c_str());
    normalEstimator.setRadiusSearch(neighborRadius);
    normalEstimator.compute(*cloudNormals);
    // /*output normals' value*/
    // for (size_t i = 0; i < cloudNormals->size(); i++) {
    //   ROS_INFO_STREAM("normal  " << i << "is: " <<
    //   cloudNormals->at(i).normal_x
    //                              << "," << cloudNormals->at(i).normal_y <<
    //                              ","
    //                              << cloudNormals->at(i).normal_z);
    // }
    /**filter normals*/
    double unit_normal_z[3] = {0, 1.0, 0};
    vector<int> normal_indices;
    for (size_t i = 0; i < cloudNormals->size(); i++) {
      double dot_product = unit_normal_z[0] * cloudNormals->at(i).normal_x +
                           unit_normal_z[1] * cloudNormals->at(i).normal_y +
                           unit_normal_z[2] * cloudNormals->at(i).normal_z;
      // ROS_INFO_STREAM("dot_product " << i << "is :" << dot_product);
      if (fabs(dot_product) < 0.5)
        normal_indices.push_back(i);
    }
    pcl::PointCloud<pcl::Normal>::Ptr filtered_normals(
        new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr filtered_frontier(new pcl::PointCloud<PointT>);
    for (size_t i = 0; i < normal_indices.size(); i++) {
      filtered_normals->push_back(cloudNormals->at(normal_indices.at(i)));
      filtered_frontier->push_back(frontierCloud->at(normal_indices.at(i)));
    }
    cout << "frontier points' size: " << frontierCloud->points.size() << endl;
    ROS_INFO_STREAM("filtered_normals size is :" << filtered_normals->size());
    ROS_INFO_STREAM("filtered_frontier size is :" << filtered_frontier->size());
    /**use kmeans cluster to classify different surface*/
    int cluster_count = 4;
    cv::Mat cv_filtered_normals(filtered_normals->size(), 3, CV_32F);
    // cout << cv_filtered_normals << endl;
    cv::Mat labels;
    cv::Mat normal_centers;
    for (size_t i = 0; i < filtered_normals->size(); i++) {
      ROS_INFO_STREAM("converting normal" << i);
      cv_filtered_normals.at<float>(i, 0) = filtered_normals->at(i).normal_x;
      cv_filtered_normals.at<float>(i, 1) = filtered_normals->at(i).normal_y;
      cv_filtered_normals.at<float>(i, 2) = filtered_normals->at(i).normal_z;
      ROS_INFO_STREAM("normal " << i << "is "
                                << cv_filtered_normals.at<float>(i, 0) << ","
                                << cv_filtered_normals.at<float>(i, 1) << ","
                                << cv_filtered_normals.at<float>(i, 2));
    }
    // cout << cv_filtered_normals << endl;
    ROS_INFO_STREAM("begin kmeans");
    // ros::Duration(30.0).sleep();

    cv::kmeans(cv_filtered_normals, cluster_count, labels,
               cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 5,
               cv::KMEANS_PP_CENTERS, normal_centers);
    /**merge similar normal and cloud*/
    ROS_INFO_STREAM("merge similar normals");
    vector<vector<int>> distinct_normal_indices;
    vector<int> first_indice;
    first_indice.push_back(0);
    distinct_normal_indices.push_back(first_indice);
    for (size_t i = 1; i < cluster_count; i++) {
      bool is_distinct = true;
      for (size_t j = 0; j < distinct_normal_indices.size(); j++) {
        int id = distinct_normal_indices.at(j).at(0);
        ROS_INFO_STREAM("id is" << id);
        double dot_product =
            normal_centers.at<float>(id, 0) * normal_centers.at<float>(i, 0) +
            normal_centers.at<float>(id, 1) * normal_centers.at<float>(i, 1) +
            normal_centers.at<float>(id, 2) * normal_centers.at<float>(i, 2);
        if (dot_product > 0.7) {
          // a similar normal
          distinct_normal_indices.at(j).push_back(i);
          is_distinct = false;
          break;
        } else {
          // a distintc normal,continue to compare with other
        }
      }
      if (is_distinct) {
        vector<int> distinct_normal_indice;
        distinct_normal_indice.push_back(i);
        distinct_normal_indices.push_back(distinct_normal_indice);
      }
    }
    for (size_t i = 0; i < distinct_normal_indices.size(); i++) {
      ROS_INFO_STREAM("cluster: " << i);
      for (size_t j = 0; j < distinct_normal_indices.at(i).size(); j++) {
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
    //                 << count_labels[0] << "," << count_labels[1] << ","
    //                 << count_labels[2] << "," << count_labels[3]);
    // for (size_t i = 0; i < filtered_normals->size(); i++) {
    //   // ROS_INFO_STREAM("normal " << i << "'s label is:" <<
    //   labels.at<int>(i));
    // }
    /*seperate different groups of normals*/
    vector<pcl::PointCloud<pcl::Normal>::Ptr> final_cluster_normals;
    vector<pcl::PointCloud<PointT>::Ptr> final_cluster_cloud;
    for (size_t i = 0; i < distinct_normal_indices.size(); i++) {
      pcl::PointCloud<pcl::Normal>::Ptr normals_ptr(
          new pcl::PointCloud<pcl::Normal>);
      final_cluster_normals.push_back(normals_ptr);
      pcl::PointCloud<PointT>::Ptr cloud_ptr(new pcl::PointCloud<PointT>);
      final_cluster_cloud.push_back(cloud_ptr);
    }
    for (size_t i = 0; i < filtered_normals->size(); i++) {
      int cluster_id = -1;
      for (size_t j = 0; j < distinct_normal_indices.size(); j++) {
        bool is_member = false;
        for (size_t k = 0; k < distinct_normal_indices.at(j).size(); k++) {
          if (labels.at<int>(i, 0) == distinct_normal_indices.at(j).at(k)) {
            is_member = true;
          }
        }
        if (is_member) {
          cluster_id = j;
          break;
        }
      }
      final_cluster_normals.at(cluster_id)->push_back(filtered_normals->at(i));
      final_cluster_cloud.at(cluster_id)->push_back(filtered_frontier->at(i));
    }
    for (size_t i = 0; i < final_cluster_normals.size(); i++) {
      ROS_INFO_STREAM("normal cluster " << i << " has "
                                        << final_cluster_normals.at(i)->size()
                                        << "normals");
      ROS_INFO_STREAM("cloud cluster " << i << " has "
                                       << final_cluster_cloud.at(i)->size()
                                       << "points");
    }
    /**visualize cluster normal and cloud*/
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    int rgb_array[4][3] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}};
    /*add different cluster of normals and cloud*/
    for (size_t i = 0; i < final_cluster_normals.size(); i++) {
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
          rgb_handler(final_cluster_cloud.at(i), rgb_array[i][0],
                      rgb_array[i][1], rgb_array[i][2]);
      viewer.addPointCloud<pcl::PointXYZRGB>(final_cluster_cloud.at(i),
                                             rgb_handler, "cloud" + i);
      viewer.setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud" + i);
      viewer.addPointCloudNormals<PointT, pcl::Normal>(
          final_cluster_cloud.at(i), final_cluster_normals.at(i), 10, 0.2,
          "normal" + i);
    }
    viewer.addCoordinateSystem(1.0);
    /**visualize origin normal and cloud*/
    pcl::visualization::PCLVisualizer viewer_o("PCL Viewer Origin");
    viewer_o.setBackgroundColor(0.0, 0.0, 0.0);
    /*add origin normals and cloud*/
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB>
        rgb_handler_o(frontierCloud, 255, 255, 0);
    viewer_o.addPointCloud<pcl::PointXYZRGB>(frontierCloud, rgb_handler_o,
                                             "cloud");
    viewer_o.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
    // viewer_o.addPointCloudNormals<PointT, pcl::Normal>(
    //     filtered_frontier, filtered_normals, 10, 0.2, "normal");
    viewer_o.addCoordinateSystem(1.0);
    while (!viewer.wasStopped()) {
      viewer.spinOnce();
    }
  }

  return 0;
}

FRAME
readFrame(int index, ParameterReader &pd) {
  FRAME f;
  string rgbDir = pd.getData("rgb_dir");
  string depthDir = pd.getData("depth_dir");

  string rgbExt = pd.getData("rgb_extension");
  string depthExt = pd.getData("depth_extension");

  stringstream ss;
  ss << rgbDir << index << rgbExt;
  string filename;
  ss >> filename;
  f.rgb = cv::imread(filename);

  ss.clear();
  filename.clear();
  ss << depthDir << index << "_depth" << depthExt;
  ss >> filename;

  f.depth = cv::imread(filename, -1);
  f.frameID = index;
  return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec) {
  return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) +
         fabs(cv::norm(tvec));
}

CHECK_RESULT
checkKeyframes(FRAME &f1, FRAME &f2, g2o::SparseOptimizer &opti,
               bool is_loops) {
  static ParameterReader pd;
  static int min_inliers = atoi(pd.getData("min_inliers").c_str());
  static double max_norm = atof(pd.getData("max_norm").c_str());
  static double keyframe_threshold =
      atof(pd.getData("keyframe_threshold").c_str());
  static double max_norm_lp = atof(pd.getData("max_norm_lp").c_str());
  static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
  // 比较f1 和 f2
  RESULT_OF_PNP result = estimateMotion(f1, f2, camera);
  if (result.inliers < min_inliers) // inliers不够，放弃该帧
    return NOT_MATCHED;
  // 计算运动范围是否太大
  double norm = normofTransform(result.rvec, result.tvec);
  if (is_loops == false) {
    if (norm >= max_norm)
      return TOO_FAR_AWAY; // too far away, may be error
  } else {
    if (norm >= max_norm_lp)
      return TOO_FAR_AWAY;
  }

  if (norm <= keyframe_threshold)
    return TOO_CLOSE; // too adjacent frame
  // 向g2o中增加这个顶点与上一帧联系的边
  // 顶点部分
  // 顶点只需设定id即可
  if (is_loops == false) {
    /*add new vertex to pose graph*/
    g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId(f2.frameID);
    v->setEstimate(Eigen::Isometry3d::Identity());
    opti.addVertex(v);
    /* add camera position to frame*/
    f2.cameraPos[0] = result.tvec.at<double>(0, 0) + f1.cameraPos[0];
    f2.cameraPos[1] = result.tvec.at<double>(1, 0) + f1.cameraPos[1];
    f2.cameraPos[2] = result.tvec.at<double>(2, 0) + f1.cameraPos[2];
  }
  // 边部分
  g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
  // 连接此边的两个顶点id
  edge->setVertex(0, opti.vertex(f1.frameID));
  edge->setVertex(1, opti.vertex(f2.frameID));
  edge->setRobustKernel(new g2o::RobustKernelHuber());
  // 信息矩阵
  Eigen::Matrix<double, 6, 6> information =
      Eigen::Matrix<double, 6, 6>::Identity();
  // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
  // 因为pose为6D的，信息矩阵是6*6的阵，假设位置
  //和角度的估计精度均为0.1且互相独立
  // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
  information(0, 0) = information(1, 1) = information(2, 2) = 100;
  information(3, 3) = information(4, 4) = information(5, 5) = 100;
  // 也可以将角度设大一些，表示对角度的估计更加准确
  edge->setInformation(information);
  // 边的估计即是pnp求解之结果
  Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
  // edge->setMeasurement( T );
  edge->setMeasurement(T.inverse());
  // 将此边加入图中
  opti.addEdge(edge);
  return KEYFRAME;
}

void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame,
                      g2o::SparseOptimizer &opti) {
  static ParameterReader pd;
  static int nearby_loops = atoi(pd.getData("nearby_loops").c_str());

  // 就是把currFrame和 frames里末尾几个测一遍
  if (frames.size() <= nearby_loops) {
    // no enough keyframes, check everyone
    for (size_t i = 0; i < frames.size(); i++) {
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  } else {
    // check the nearest ones
    for (size_t i = frames.size() - nearby_loops; i < frames.size(); i++) {
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
}

void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame,
                      g2o::SparseOptimizer &opti) {
  static ParameterReader pd;
  static int random_loops = atoi(pd.getData("random_loops").c_str());
  srand((unsigned int)time(NULL));
  // 随机取一些帧进行检测

  if (frames.size() <= random_loops) {
    // no enough keyframes, check everyone
    for (size_t i = 0; i < frames.size(); i++) {
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  } else {
    // randomly check loops
    for (int i = 0; i < random_loops; i++) {
      int index = rand() % frames.size();
      checkKeyframes(frames[index], currFrame, opti, true);
    }
  }
}

PointCloud::Ptr image2PointCloud(cv::Mat &rgb, cv::Mat &depth,
                                 CAMERA_INTRINSIC_PARAMETERS &camera) {
  PointCloud::Ptr cloud(new PointCloud);

  for (int m = 0; m < depth.rows; m += 2)
    for (int n = 0; n < depth.cols; n += 2) {
      // 获取深度图中(m,n)处的值
      ushort d = depth.ptr<ushort>(m)[n];
      // d 可能没有值，若如此，跳过此点
      if (d == 0)
        continue;
      // d 存在值，则向点云增加一个点
      PointT p;

      // 计算这个点的空间坐标
      p.z = double(d) / camera.scale;
      p.x = (n - camera.cx) * p.z / camera.fx;
      p.y = (m - camera.cy) * p.z / camera.fy;

      // 从rgb图像中获取它的颜色
      // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
      p.b = rgb.ptr<uchar>(m)[n * 3];
      p.g = rgb.ptr<uchar>(m)[n * 3 + 1];
      p.r = rgb.ptr<uchar>(m)[n * 3 + 2];

      // 把p加入到点云中
      cloud->points.push_back(p);
    }
  // 设置并保存点云
  cloud->height = 1;
  cloud->width = cloud->points.size();
  cloud->is_dense = false;

  return cloud;
}

cv::Point3f point2dTo3d(cv::Point3f &point,
                        CAMERA_INTRINSIC_PARAMETERS &camera) {
  cv::Point3f p; // 3D 点
  p.z = double(point.z) / camera.scale;
  p.x = (point.x - camera.cx) * p.z / camera.fx;
  p.y = (point.y - camera.cy) * p.z / camera.fy;
  return p;
}

// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void computeKeyPointsAndDesp(FRAME &frame, string detector, string descriptor) {
  cv::Ptr<cv::FeatureDetector> _detector;
  cv::Ptr<cv::DescriptorExtractor> _descriptor;
  cv::initModule_nonfree();
  _detector = cv::FeatureDetector::create(detector.c_str());
  _descriptor = cv::DescriptorExtractor::create(descriptor.c_str());

  if (!_detector || !_descriptor) {
    cerr << "Unknown detector or discriptor type !" << detector << ","
         << descriptor << endl;
    return;
  }
  _detector->detect(frame.rgb, frame.kp);
  _descriptor->compute(frame.rgb, frame.kp, frame.desp);
  // show descriptor
  cv::Mat imgShow;
  cv::drawKeypoints(frame.rgb, frame.kp, imgShow, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("key points", imgShow);
  cv::waitKey(1);

  return;
}

// estimateMotion 计算两个帧之间的运动
// 输入：帧1和帧2
// 输出：rvec 和 tvec
RESULT_OF_PNP
estimateMotion(FRAME &frame1, FRAME &frame2,
               CAMERA_INTRINSIC_PARAMETERS &camera) {
  static ParameterReader pd;
  vector<cv::DMatch> matches;
  cv::BFMatcher matcher;
  matcher.match(frame1.desp, frame2.desp, matches);

  RESULT_OF_PNP result;
  vector<cv::DMatch> goodMatches;
  double minDis = 9999;
  double good_match_threshold =
      atof(pd.getData("good_match_threshold").c_str());
  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i].distance < minDis)
      minDis = matches[i].distance;
  }

  if (minDis < 10)
    minDis = 10;

  for (size_t i = 0; i < matches.size(); i++) {
    if (matches[i].distance < good_match_threshold * minDis)
      goodMatches.push_back(matches[i]);
  }
  // show matches
  cv::Mat imgMatches;
  cv::drawMatches(frame1.rgb, frame1.kp, frame2.rgb, frame2.kp, goodMatches,
                  imgMatches);
  cv::imshow("good matches", imgMatches);
  cv::waitKey(1);

  if (goodMatches.size() <= 5) {
    result.inliers = -1;
    return result;
  }
  // 第一个帧的三维点
  vector<cv::Point3f> pts_obj;
  // 第二个帧的图像点
  vector<cv::Point2f> pts_img;

  // 相机内参
  for (size_t i = 0; i < goodMatches.size(); i++) {
    // query 是第一个, train 是第二个
    cv::Point2f p = frame1.kp[goodMatches[i].queryIdx].pt;
    // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
    ushort d = frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
    if (d == 0)
      continue;
    pts_img.push_back(cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));

    // 将(u,v,d)转成(x,y,z)
    cv::Point3f pt(p.x, p.y, d);
    cv::Point3f pd = point2dTo3d(pt, camera);
    pts_obj.push_back(pd);
  }

  if (pts_obj.size() == 0 || pts_img.size() == 0) {
    result.inliers = -1;
    return result;
  }

  double camera_matrix_data[3][3] = {
      {camera.fx, 0, camera.cx}, {0, camera.fy, camera.cy}, {0, 0, 1}};

  // 构建相机矩阵
  cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
  cv::Mat rvec, tvec, inliers;
  // 求解pnp
  cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec, tvec,
                     false, 100, 1.0, 100, inliers);

  result.rvec = rvec;
  result.tvec = tvec;
  // cout<<" T is :"<<tvec<<endl;
  // cout<<tvec.at<double>(0,0)<<" "<<tvec.at<double>(1,0)<<"
  // "<<tvec.at<double>(2,0)<<endl;
  result.inliers = inliers.rows;

  return result;
}

// cvMat2Eigen
Eigen::Isometry3d cvMat2Eigen(cv::Mat &rvec, cv::Mat &tvec) {
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  Eigen::Matrix3d r;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      r(i, j) = R.at<double>(i, j);

  // 将平移向量和旋转矩阵转换成变换矩阵
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

  Eigen::AngleAxisd angle(r);
  T = angle;
  T(0, 3) = tvec.at<double>(0, 0);
  T(1, 3) = tvec.at<double>(1, 0);
  T(2, 3) = tvec.at<double>(2, 0);
  return T;
}

// joinPointCloud
// 输入：原始点云，新来的帧以及它的位姿
// 输出：将新来帧加到原始帧后的图像
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original, FRAME &newFrame,
                               Eigen::Isometry3d T,
                               CAMERA_INTRINSIC_PARAMETERS &camera) {
  PointCloud::Ptr newCloud =
      image2PointCloud(newFrame.rgb, newFrame.depth, camera);

  // 合并点云
  PointCloud::Ptr output(new PointCloud());
  pcl::transformPointCloud(*original, *output, T.matrix());
  *newCloud += *output;

  // Voxel grid 滤波降采样
  static pcl::VoxelGrid<PointT> voxel;
  static ParameterReader pd;
  double gridsize = atof(pd.getData("voxel_grid").c_str());
  voxel.setLeafSize(gridsize, gridsize, gridsize);
  voxel.setInputCloud(newCloud);
  PointCloud::Ptr tmp(new PointCloud());
  voxel.filter(*tmp);
  return tmp;
}

CAMERA_INTRINSIC_PARAMETERS getDefaultCamera() {
  ParameterReader pd;
  CAMERA_INTRINSIC_PARAMETERS camera;
  camera.fx = atof(pd.getData("camera.fx").c_str());
  camera.fy = atof(pd.getData("camera.fy").c_str());
  camera.cx = atof(pd.getData("camera.cx").c_str());
  camera.cy = atof(pd.getData("camera.cy").c_str());
  camera.scale = atof(pd.getData("camera.scale").c_str());
  return camera;
}

void rgbImageCallback(const sensor_msgs::Image::ConstPtr msg)
{    
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg,
            sensor_msgs::image_encodings::BGR8);
    //cv::imshow(wndname,cv_ptr->image);
    cv::Mat image = cv_bridge::toCvCopy(
        msg,sensor_msgs::image_encodings::BGR8 )->image;
    cv::Mat image_p;
    image.copyTo(image_p);    
}

void depthImageCallback(const sensor_msgs::Image::ConstPtr msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg,
            sensor_msgs::image_encodings::BGR8);
    //cv::imshow(wndname,cv_ptr->image);
    cv::Mat image = cv_bridge::toCvCopy(
        msg,sensor_msgs::image_encodings::BGR8 )->image;
    cv::Mat image_p;
    image.copyTo(image_p);   
}

void print_query_info(point3d query, ColorOcTreeNode* node) {
  if (node != NULL) 
  {
        std::cout <<query<<"\t"<< "Occupancy is: " <<node->getOccupancy()<<endl
                             <<" logOdds is :" <<node->getLogOdds()<<endl
                             <<" Value is: "<<node->getValue()<<endl;
  }
  else 
    cout << "occupancy probability at " << query << ":\t is unknown" << endl;    
}
