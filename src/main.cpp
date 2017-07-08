#include "drone_explorer.h"

int main(int argc, char const *argv[])
{
  DroneExplorer droneExplorer;
  ParameterReader parameter_reader;
  int mode= atoi(parameter_reader.getData("task_mode").c_str());
  switch(mode)
  {
    case 0:
    {
      int start_index=
          atoi(parameter_reader.getData("start_index").c_str());
      int end_index=
          atoi(parameter_reader.getData("end_index").c_str());

      droneExplorer.setStartIndex(start_index);
      droneExplorer.initialize();
      /*set the first key frame*/
      rgbd::FRAME frame=
          rgbd::readFrame(start_index, parameter_reader);
      droneExplorer.computeKeyPointsAndDesp(frame);
      frame.cameraPos[0]= frame.cameraPos[1]= frame.cameraPos[2]= 0.0;
      droneExplorer.addNewKeyFrame(frame);

      /*loop for every new frame*/
      for(int i= start_index + 1; i < end_index; ++i)
      {
        frame= rgbd::readFrame(i, parameter_reader);
        droneExplorer.processNewFrame(frame);
      }
      droneExplorer.saveOctreeMap(
          "/home/zby/uav_slam_ws/src/rgbd_slam/"
          "generated/global_octree.ot");
      droneExplorer.savePointCloudMap(
          "/home/zby/uav_slam_ws/src/rgbd_slam/"
          "generated/global_pointcloud.pcd");
    }
    break;

    case 1:
    {
      droneExplorer.initialize();
      droneExplorer.loadOctreeMap("/home/zby/uav_slam_ws/src/"
                                  "rgbd_slam/generated/"
                                  "global_octree.ot");

      /*update frontier based on current octree map*/
      droneExplorer.updateFrontierPoints();

      /*find the target frontier and visualize it*/
      droneExplorer.searchTargetFrontier();
    }
    break;

    default:
      break;
  }

  return 0;
}