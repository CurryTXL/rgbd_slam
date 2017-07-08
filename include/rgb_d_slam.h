/**
* @file
*   This file contains api for rgb-d slam
* including visual odometry, mapping, loop enclosure
* and graph optimization.
*/

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

// C++标准库
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
using namespace std;

class ParameterReader;

namespace rgbd
{
/**相机内参结构*/
struct CAMERA_INTRINSIC_PARAMETERS
{
  double cx, cy, fx, fy, scale;
};

/**frame structure*/
struct FRAME
{
  int frameID;
  cv::Mat rgb, depth;       //该帧对应的彩色图与深度图
  cv::Mat desp;             //特征描述子
  vector<cv::KeyPoint> kp;  //关键点
  double cameraPos[3];      // current camera position
};

/*PnP 结果*/
struct RESULT_OF_PNP
{
  cv::Mat rvec, tvec;
  int inliers;
};

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType>
    SlamLinearSolver;

FRAME readFrame(int index, ParameterReader &pd);

/**估计一个运动的大小
* @param rvec rotation vector
* @param tvec translation vector
* @return a value represent how far the
*   motion is
*/
double normofTransform(cv::Mat rvec, cv::Mat tvec);

/**检测两个帧，结果定义*/
enum CHECK_RESULT
{
  NOT_MATCHED= 0,
  TOO_FAR_AWAY,
  TOO_CLOSE,
  KEYFRAME
};
/**检查是否为关键帧*/
CHECK_RESULT checkKeyframes(FRAME &f1, FRAME &f2,
                            g2o::SparseOptimizer &opti,
                            bool is_loops= false);

/**检测近距离的回环*/
void checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame,
                      g2o::SparseOptimizer &opti);

/**随机检测回环*/
void checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame,
                      g2o::SparseOptimizer &opti);

/**将rgb图转换为点云*/
PointCloud::Ptr image2PointCloud(cv::Mat &rgb, cv::Mat &depth,
                                 CAMERA_INTRINSIC_PARAMETERS &camera);

/**point2dTo3d 将单个点从图像坐标转换为空间坐标*/
cv::Point3f point2dTo3d(cv::Point3f &point,
                        CAMERA_INTRINSIC_PARAMETERS &camera);

/**提取关键点与特征描述子*/
void computeKeyPointsAndDesp(FRAME &frame, string detector,
                             string descriptor);

/**estimateMotion 计算两个帧之间的运动
*输入：帧1和帧2, 相机内参
*/
RESULT_OF_PNP estimateMotion(FRAME &frame1, FRAME &frame2,
                             CAMERA_INTRINSIC_PARAMETERS &camera);

/**convert opencv Mat to Eigen Mat*/
Eigen::Isometry3d cvMat2Eigen(cv::Mat &rvec, cv::Mat &tvec);

/**
*输入：原始点云，新来的帧以及它的位姿
* 输出：将新来帧加到原始帧后的图像
*/
PointCloud::Ptr joinPointCloud(PointCloud::Ptr original,
                               FRAME &newFrame, Eigen::Isometry3d T,
                               CAMERA_INTRINSIC_PARAMETERS &camera);

/**get camera intristic parameter*/
CAMERA_INTRINSIC_PARAMETERS getDefaultCamera();
}
