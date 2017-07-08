#include "rgb_d_slam.h"
#include "parameter_reader.h"
using namespace rgbd;

double rgbd::normofTransform(cv::Mat rvec, cv::Mat tvec)
{
  return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) +
         fabs(cv::norm(tvec));
}

CHECK_RESULT rgbd::checkKeyframes(FRAME &f1, FRAME &f2,
                                  g2o::SparseOptimizer &opti,
                                  bool is_loops)
{
  static ParameterReader pd;
  static int min_inliers= atoi(pd.getData("min_inliers").c_str());
  static double max_norm= atof(pd.getData("max_norm").c_str());
  static double keyframe_threshold=
      atof(pd.getData("keyframe_threshold").c_str());
  static double max_norm_lp= atof(pd.getData("max_norm_lp").c_str());
  static CAMERA_INTRINSIC_PARAMETERS camera= getDefaultCamera();
  // 比较f1 和 f2
  RESULT_OF_PNP result= estimateMotion(f1, f2, camera);
  if(result.inliers < min_inliers)  // inliers不够，放弃该帧
    return NOT_MATCHED;
  // 计算运动范围是否太大
  double norm= normofTransform(result.rvec, result.tvec);
  if(is_loops == false)
  {
    if(norm >= max_norm)
      return TOO_FAR_AWAY;  // too far away, may be error
  }
  else
  {
    if(norm >= max_norm_lp)
      return TOO_FAR_AWAY;
  }

  if(norm <= keyframe_threshold)
    return TOO_CLOSE;  // too adjacent frame
  // 向g2o中增加这个顶点与上一帧联系的边
  // 顶点部分
  // 顶点只需设定id即可
  if(is_loops == false)
  {
    /*add new vertex to pose graph*/
    g2o::VertexSE3 *v= new g2o::VertexSE3();
    v->setId(f2.frameID);
    v->setEstimate(Eigen::Isometry3d::Identity());
    opti.addVertex(v);
    /* add camera position to frame*/
    f2.cameraPos[0]= result.tvec.at<double>(0, 0) + f1.cameraPos[0];
    f2.cameraPos[1]= result.tvec.at<double>(1, 0) + f1.cameraPos[1];
    f2.cameraPos[2]= result.tvec.at<double>(2, 0) + f1.cameraPos[2];
  }
  // 边部分
  g2o::EdgeSE3 *edge= new g2o::EdgeSE3();
  // 连接此边的两个顶点id
  edge->setVertex(0, opti.vertex(f1.frameID));
  edge->setVertex(1, opti.vertex(f2.frameID));
  edge->setRobustKernel(new g2o::RobustKernelHuber());
  // 信息矩阵
  Eigen::Matrix<double, 6, 6> information=
      Eigen::Matrix<double, 6, 6>::Identity();
  // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
  // 因为pose为6D的，信息矩阵是6*6的阵，假设位置
  //和角度的估计精度均为0.1且互相独立
  // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
  information(0, 0)= information(1, 1)= information(2, 2)= 100;
  information(3, 3)= information(4, 4)= information(5, 5)= 100;
  // 也可以将角度设大一些，表示对角度的估计更加准确
  edge->setInformation(information);
  // 边的估计即是pnp求解之结果
  Eigen::Isometry3d T= cvMat2Eigen(result.rvec, result.tvec);
  // edge->setMeasurement( T );
  edge->setMeasurement(T.inverse());
  // 将此边加入图中
  opti.addEdge(edge);
  return KEYFRAME;
}

void rgbd::checkNearbyLoops(vector<FRAME> &frames, FRAME &currFrame,
                            g2o::SparseOptimizer &opti)
{
  static ParameterReader pd;
  static int nearby_loops= atoi(pd.getData("nearby_loops").c_str());

  // 就是把currFrame和 frames里末尾几个测一遍
  if(frames.size() <= nearby_loops)
  {
    // no enough keyframes, check everyone
    for(size_t i= 0; i < frames.size(); i++)
    {
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
  else
  {
    // check the nearest ones
    for(size_t i= frames.size() - nearby_loops; i < frames.size();
        i++)
    {
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
}

void rgbd::checkRandomLoops(vector<FRAME> &frames, FRAME &currFrame,
                            g2o::SparseOptimizer &opti)
{
  static ParameterReader pd;
  static int random_loops= atoi(pd.getData("random_loops").c_str());
  srand((unsigned int)time(NULL));
  // 随机取一些帧进行检测

  if(frames.size() <= random_loops)
  {
    // no enough keyframes, check everyone
    for(size_t i= 0; i < frames.size(); i++)
    {
      checkKeyframes(frames[i], currFrame, opti, true);
    }
  }
  else
  {
    // randomly check loops
    for(int i= 0; i < random_loops; i++)
    {
      int index= rand() % frames.size();
      checkKeyframes(frames[index], currFrame, opti, true);
    }
  }
}

PointCloud::Ptr rgbd::image2PointCloud(
    cv::Mat &rgb, cv::Mat &depth, CAMERA_INTRINSIC_PARAMETERS &camera)
{
  PointCloud::Ptr cloud(new PointCloud);

  for(int m= 0; m < depth.rows; m+= 2)
    for(int n= 0; n < depth.cols; n+= 2)
    {
      // 获取深度图中(m,n)处的值
      ushort d= depth.ptr<ushort>(m)[n];
      // d 可能没有值，若如此，跳过此点
      if(d == 0)
        continue;
      // d 存在值，则向点云增加一个点
      PointT p;

      // 计算这个点的空间坐标
      p.z= double(d) / camera.scale;
      p.x= (n - camera.cx) * p.z / camera.fx;
      p.y= (m - camera.cy) * p.z / camera.fy;

      // 从rgb图像中获取它的颜色
      // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
      p.b= rgb.ptr<uchar>(m)[n * 3];
      p.g= rgb.ptr<uchar>(m)[n * 3 + 1];
      p.r= rgb.ptr<uchar>(m)[n * 3 + 2];

      // 把p加入到点云中
      cloud->points.push_back(p);
    }
  // 设置并保存点云
  cloud->height= 1;
  cloud->width= cloud->points.size();
  cloud->is_dense= false;

  return cloud;
}

cv::Point3f rgbd::point2dTo3d(cv::Point3f &point,
                              CAMERA_INTRINSIC_PARAMETERS &camera)
{
  cv::Point3f p;  // 3D 点
  p.z= double(point.z) / camera.scale;
  p.x= (point.x - camera.cx) * p.z / camera.fx;
  p.y= (point.y - camera.cy) * p.z / camera.fy;
  return p;
}

// computeKeyPointsAndDesp 同时提取关键点与特征描述子
void rgbd::computeKeyPointsAndDesp(FRAME &frame, string detector,
                                   string descriptor)
{
  cv::Ptr<cv::FeatureDetector> _detector;
  cv::Ptr<cv::DescriptorExtractor> _descriptor;
  cv::initModule_nonfree();
  _detector= cv::FeatureDetector::create(detector.c_str());
  _descriptor= cv::DescriptorExtractor::create(descriptor.c_str());

  if(!_detector || !_descriptor)
  {
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
RESULT_OF_PNP rgbd::estimateMotion(
    FRAME &frame1, FRAME &frame2, CAMERA_INTRINSIC_PARAMETERS &camera)
{
  static ParameterReader pd;
  vector<cv::DMatch> matches;
  cv::BFMatcher matcher;
  matcher.match(frame1.desp, frame2.desp, matches);

  RESULT_OF_PNP result;
  vector<cv::DMatch> goodMatches;
  double minDis= 9999;
  double good_match_threshold=
      atof(pd.getData("good_match_threshold").c_str());
  for(size_t i= 0; i < matches.size(); i++)
  {
    if(matches[i].distance < minDis)
      minDis= matches[i].distance;
  }

  if(minDis < 10)
    minDis= 10;

  for(size_t i= 0; i < matches.size(); i++)
  {
    if(matches[i].distance < good_match_threshold * minDis)
      goodMatches.push_back(matches[i]);
  }
  // show matches
  cv::Mat imgMatches;
  cv::drawMatches(frame1.rgb, frame1.kp, frame2.rgb, frame2.kp,
                  goodMatches, imgMatches);
  cv::imshow("good matches", imgMatches);
  cv::waitKey(1);

  if(goodMatches.size() <= 5)
  {
    result.inliers= -1;
    return result;
  }
  // 第一个帧的三维点
  vector<cv::Point3f> pts_obj;
  // 第二个帧的图像点
  vector<cv::Point2f> pts_img;

  // 相机内参
  for(size_t i= 0; i < goodMatches.size(); i++)
  {
    // query 是第一个, train 是第二个
    cv::Point2f p= frame1.kp[goodMatches[i].queryIdx].pt;
    // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
    ushort d= frame1.depth.ptr<ushort>(int(p.y))[int(p.x)];
    if(d == 0)
      continue;
    pts_img.push_back(
        cv::Point2f(frame2.kp[goodMatches[i].trainIdx].pt));

    // 将(u,v,d)转成(x,y,z)
    cv::Point3f pt(p.x, p.y, d);
    cv::Point3f pd= point2dTo3d(pt, camera);
    pts_obj.push_back(pd);
  }

  if(pts_obj.size() == 0 || pts_img.size() == 0)
  {
    result.inliers= -1;
    return result;
  }

  double camera_matrix_data[3][3]= { { camera.fx, 0, camera.cx },
                                     { 0, camera.fy, camera.cy },
                                     { 0, 0, 1 } };

  // 构建相机矩阵
  cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
  cv::Mat rvec, tvec, inliers;
  // 求解pnp
  cv::solvePnPRansac(pts_obj, pts_img, cameraMatrix, cv::Mat(), rvec,
                     tvec, false, 100, 1.0, 100, inliers);

  result.rvec= rvec;
  result.tvec= tvec;
  // cout<<" T is :"<<tvec<<endl;
  // cout<<tvec.at<double>(0,0)<<" "<<tvec.at<double>(1,0)<<"
  // "<<tvec.at<double>(2,0)<<endl;
  result.inliers= inliers.rows;

  return result;
}

// cvMat2Eigen
Eigen::Isometry3d rgbd::cvMat2Eigen(cv::Mat &rvec, cv::Mat &tvec)
{
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  Eigen::Matrix3d r;
  for(int i= 0; i < 3; i++)
    for(int j= 0; j < 3; j++)
      r(i, j)= R.at<double>(i, j);

  // 将平移向量和旋转矩阵转换成变换矩阵
  Eigen::Isometry3d T= Eigen::Isometry3d::Identity();

  Eigen::AngleAxisd angle(r);
  T= angle;
  T(0, 3)= tvec.at<double>(0, 0);
  T(1, 3)= tvec.at<double>(1, 0);
  T(2, 3)= tvec.at<double>(2, 0);
  return T;
}

PointCloud::Ptr rgbd::joinPointCloud(
    PointCloud::Ptr original, FRAME &newFrame, Eigen::Isometry3d T,
    CAMERA_INTRINSIC_PARAMETERS &camera)
{
  PointCloud::Ptr newCloud=
      image2PointCloud(newFrame.rgb, newFrame.depth, camera);

  // 合并点云
  PointCloud::Ptr output(new PointCloud());
  pcl::transformPointCloud(*original, *output, T.matrix());
  *newCloud+= *output;

  // Voxel grid 滤波降采样
  static pcl::VoxelGrid<PointT> voxel;
  static ParameterReader pd;
  double gridsize= atof(pd.getData("voxel_grid").c_str());
  voxel.setLeafSize(gridsize, gridsize, gridsize);
  voxel.setInputCloud(newCloud);
  PointCloud::Ptr tmp(new PointCloud());
  voxel.filter(*tmp);
  return tmp;
}

CAMERA_INTRINSIC_PARAMETERS rgbd::getDefaultCamera()
{
  ParameterReader pd;
  CAMERA_INTRINSIC_PARAMETERS camera;
  camera.fx= atof(pd.getData("camera.fx").c_str());
  camera.fy= atof(pd.getData("camera.fy").c_str());
  camera.cx= atof(pd.getData("camera.cx").c_str());
  camera.cy= atof(pd.getData("camera.cy").c_str());
  camera.scale= atof(pd.getData("camera.scale").c_str());
  return camera;
}

// 给定index，读取一帧数据
FRAME rgbd::readFrame(int index, ParameterReader &pd)
{
  FRAME f;
  string rgbDir= pd.getData("rgb_dir");
  string depthDir= pd.getData("depth_dir");

  string rgbExt= pd.getData("rgb_extension");
  string depthExt= pd.getData("depth_extension");

  stringstream ss;
  ss << rgbDir << index << rgbExt;
  string filename;
  ss >> filename;
  f.rgb= cv::imread(filename);

  ss.clear();
  filename.clear();
  ss << depthDir << index << "_depth" << depthExt;
  ss >> filename;

  f.depth= cv::imread(filename, -1);
  f.frameID= index;
  return f;
}