#include "ros/ros.h"
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "visualization_msgs/Marker.h"
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>

#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/SimpleSetup.h>

#include <ompl/config.h>
#include <iostream>

#include "fcl/config.h"
#include "fcl/octree.h"
#include "fcl/traversal/traversal_node_octree.h"
#include "fcl/collision.h"
#include "fcl/broadphase/broadphase.h"
#include "fcl/math/transform.h"

#include <mav_trajectory_generation/polynomial_optimization_linear.h>
#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <mav_trajectory_generation/trajectory.h>
#include <mav_trajectory_generation/trajectory_sampling.h>
#include <mav_trajectory_generation_ros/ros_visualization.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// Declear some global variables
trajectory_msgs::MultiDOFJointTrajectoryPoint point_msg;
trajectory_msgs::MultiDOFJointTrajectoryPoint point_msg_prev;
// goal point
float goal_x = 0.0;
float goal_y = 0.0;
float goal_z = 0.0;
// start point
float start_x = 0.0;
float start_y = 0.0;
float start_z = 0.0;
// max planning duration
double max_planning_duration = 10.0;
// bool flag to check when to start planning
bool flag_sub_goal = 0;
// bool flag to check whether to use bounding box as SE(3) space or not
bool is_bounding_box_flag = 1;
//ROS publishers
ros::Publisher vis_pub;
ros::Publisher start_pub, goal_pub;

std::shared_ptr<fcl::CollisionGeometry> Quadcopter(new fcl::Box(0.3, 0.3, 0.1));//0.3,0.3.0.1
fcl::OcTree* tree = new fcl::OcTree(std::shared_ptr<const octomap::OcTree>(new octomap::OcTree(0.1)));
fcl::CollisionObject treeObj((std::shared_ptr<fcl::CollisionGeometry>(tree)));
fcl::CollisionObject aircraftObject(Quadcopter);

bool isStateValid(const ob::State *state)
{
    // cast the abstract state type to the type we expect
    const ob::SE3StateSpace::StateType *se3state = state->as<ob::SE3StateSpace::StateType>();

    // extract the first component of the state and cast it to what we expect
    const ob::RealVectorStateSpace::StateType *pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);

    // extract the second component of the state and cast it to what we expect
    const ob::SO3StateSpace::StateType *rot = se3state->as<ob::SO3StateSpace::StateType>(1);

    // check validity of state Fdefined by pos & rot
    fcl::Vec3f translation(pos->values[0],pos->values[1],pos->values[2]);
    fcl::Quaternion3f rotation(rot->w, rot->x, rot->y, rot->z);
    aircraftObject.setTransform(rotation, translation);
    fcl::CollisionRequest requestType(1,false,1,false);
    fcl::CollisionResult collisionResult;
    fcl::collide(&aircraftObject, &treeObj, requestType, collisionResult);

    return(!collisionResult.isCollision());
}

bool isWaypointValid(const ob::ScopedState<ob::SE3StateSpace> &se3state)
{

    // extract the first component of the state and cast it to what we expect
    const ob::RealVectorStateSpace::StateType *pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);

    // extract the second component of the state and cast it to what we expect
    const ob::SO3StateSpace::StateType *rot = se3state->as<ob::SO3StateSpace::StateType>(1);

    // check validity of state Fdefined by pos & rot
    fcl::Vec3f translation(pos->values[0],pos->values[1],pos->values[2]);
    fcl::Quaternion3f rotation(rot->w, rot->x, rot->y, rot->z);
    aircraftObject.setTransform(rotation, translation);
    fcl::CollisionRequest requestType(1,false,1,false);
    fcl::CollisionResult collisionResult;
    fcl::collide(&aircraftObject, &treeObj, requestType, collisionResult);

    return(!collisionResult.isCollision());
}

ob::OptimizationObjectivePtr getThresholdPathLengthObj(const ob::SpaceInformationPtr& si)
{
    ob::OptimizationObjectivePtr obj(new ob::PathLengthOptimizationObjective(si));
    // obj->setCostThreshold(ob::Cost(1.51));
    return obj;
}

ob::OptimizationObjectivePtr getPathLengthObjWithCostToGo(const ob::SpaceInformationPtr& si)
{
    ob::OptimizationObjectivePtr obj(new ob::PathLengthOptimizationObjective(si));
    obj->setCostToGoHeuristic(&ob::goalRegionCostToGo);
    return obj;
}

void waypointsCallback( trajectory_msgs::MultiDOFJointTrajectory msg)
{
    trajectory_msgs::MultiDOFJointTrajectoryPoint point_msg;
    int num_waypoints = int(msg.points.size());

    ROS_INFO_STREAM("######### Number of waypoints present: " << num_waypoints << "#########");

    //########define vertices and that ther are 3d
    mav_trajectory_generation::Vertex::Vector vertices;
    const int dimension = 3;
    const int derivative_to_optimize = mav_trajectory_generation::derivative_order::SNAP;
    mav_trajectory_generation::Vertex start(dimension), middle(dimension), end(dimension);

    if (num_waypoints > 1) //valid number of waypoints
    {
        for(int index=0 ; index < num_waypoints ; index++)
        {
            mav_trajectory_generation::Vertex temp_point(3);
            mav_trajectory_generation::Vertex fix_point(3);
            point_msg = msg.points[index];
            //store previous waypoint
            float prev_x = 0.0, prev_y = 0.0, prev_z = 0.0, x, y, z;
            if(index)
            {
                point_msg_prev = msg.points[index-1];
                point_msg_prev.transforms.resize(1);
                prev_x = point_msg_prev.transforms[0].translation.x;
                prev_y = point_msg_prev.transforms[0].translation.y;
                prev_z = point_msg_prev.transforms[0].translation.z;
            }

            point_msg.transforms.resize(1);
            x = point_msg.transforms[0].translation.x;
            y = point_msg.transforms[0].translation.y;
            z = point_msg.transforms[0].translation.z;

            if(index == 0)
            {
                temp_point.makeStartOrEnd(Eigen::Vector3d(x,y,z), derivative_to_optimize);
                vertices.push_back(temp_point);
            }

            else if(index == num_waypoints-1)
            {
                if(num_waypoints == 2) //add a middle point
                {
                    fix_point.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector3d((x+prev_x)/2,(y+prev_y)/2,(z+prev_z)/2));
                    vertices.push_back(fix_point);
                }
                temp_point.makeStartOrEnd(Eigen::Vector3d(x,y,z), derivative_to_optimize);
                vertices.push_back(temp_point);
            }
            else
            {
                temp_point.addConstraint(mav_trajectory_generation::derivative_order::POSITION, Eigen::Vector3d(x,y,z));
                vertices.push_back(temp_point);
            }
        }

        //##########compute the segment times

        std::vector<double> segment_times;
        const double v_max = 2.0;
        const double a_max = 2.0;
        const double magic_fabian_constant = 6.5; // A tuning parameter.
        segment_times = estimateSegmentTimes(vertices, v_max, a_max, magic_fabian_constant);

        //#######set the parameters for nonlinear Optimization

        mav_trajectory_generation::NonlinearOptimizationParameters parameters;
        parameters.max_iterations = 1000;
        parameters.f_rel = 0.05;
        parameters.x_rel = 0.1;
        parameters.time_penalty = 500.0;
        parameters.initial_stepsize_rel = 0.1;
        parameters.inequality_constraint_tolerance = 0.1;

        //#######create optimizer object and solve. (true/false) specifies if optimization run on just segment times
        const int N = 20;
        mav_trajectory_generation::PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
        opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
        opt.addMaximumMagnitudeConstraint(mav_trajectory_generation::derivative_order::VELOCITY, v_max);
        opt.optimize();

        //########obtain polynomial segments
        mav_trajectory_generation::Segment::Vector segments;
        opt.getPolynomialOptimizationRef().getSegments(&segments);

        //##########creating trajectories
        mav_trajectory_generation::Trajectory trajectory;
        opt.getTrajectory(&trajectory);

        //############sampling trajectories

        // Single sample:
        double sampling_time = 2.0;
        int derivative_order = mav_trajectory_generation::derivative_order::POSITION;
        Eigen::VectorXd sample = trajectory.evaluate(sampling_time, derivative_order);

        // Sample range:
        double t_start = 2.0;
        double t_end = 10.0;
        double dt = 0.01;
        std::vector<Eigen::VectorXd> result;
        std::vector<double> sampling_times; // Optional.
        trajectory.evaluateRange(t_start, t_end, dt, derivative_order, &result, &sampling_times);

        mav_msgs::EigenTrajectoryPoint state;
        mav_msgs::EigenTrajectoryPoint::Vector states;

        // Single sample:
        //double sampling_time = 2.0;
        bool success = mav_trajectory_generation::sampleTrajectoryAtTime(trajectory, sampling_time, &state);

        // Sample range:
        //double t_start = 2.0;
        double duration = 10.0;
        //double dt = 0.01;
        success = mav_trajectory_generation::sampleTrajectoryInRange(trajectory, t_start, duration, dt, &states);

        // Whole trajectory:
        double sampling_interval = 0.01;
        success = mav_trajectory_generation::sampleWholeTrajectory(trajectory, sampling_interval, &states);
        ROS_INFO("stamp: %d", success);

        //###########rviz vizualization
        int32_t shape = visualization_msgs::Marker::TEXT_VIEW_FACING;
        uint32_t action = 0;

        uint32_t iteration = 0;

        visualization_msgs::MarkerArray markers;
        double distance = 4.0;  // Distance by which to seperate additional markers. Set 0.0 to disable.
        std::string frame_id = "world";

        // From Trajectory class:
        mav_trajectory_generation::drawMavTrajectory(trajectory, distance, frame_id, &markers);

        // From mav_msgs::EigenTrajectoryPoint::Vector states:
        mav_trajectory_generation::drawMavSampledTrajectory(states, distance, frame_id, &markers);

        vis_pub.publish(markers);
    }
}

visualization_msgs::Marker get_extreme_points_marker_from_position(float x, float y, float z, ros::Time timestamp, bool is_goal = 0)
{
  visualization_msgs::Marker marker;

  marker.header.frame_id = "world";
  marker.header.stamp = timestamp;
  marker.ns = "path";
  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = x;
  marker.pose.position.y = y;
  marker.pose.position.z = z;
  marker.pose.orientation.x = 0;
  marker.pose.orientation.y = 0;
  marker.pose.orientation.z = 0;
  marker.pose.orientation.w = 1;
  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = 0.3;

  if(is_goal)
  {
    // if goal point
    marker.id = 1;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
  }
  else
  {
    // if start point
    marker.id = 0;
    marker.color.a = 1.0;
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
  }
  return marker;
}

void plan_path_to_goal()
{
    // construct the state space we are planning in
    ob::StateSpacePtr space(new ob::SE3StateSpace());

    // set the bounds for the R^3 part of SE(3)
    ob::RealVectorBounds bounds(3);

    if(! is_bounding_box_flag)
    {
      // for infinite bounding box
      bounds.setLow(-1);
      bounds.setHigh(1);
    }
    else
    {
      // for finite bounding area
      bounds.setLow(0,-50);
      bounds.setHigh(0,50);
      bounds.setLow(1,-50);
      bounds.setHigh(1,50);
      bounds.setLow(2,-50);
      bounds.setHigh(2,50);
    }

    space->as<ob::SE3StateSpace>()->setBounds(bounds);

    // construct an instance of  space information from this state space
    ob::SpaceInformationPtr si(new ob::SpaceInformation(space));

    // set state validity checking for this space
    si->setStateValidityChecker(std::bind(&isStateValid, std::placeholders::_1));

    // create the start state
    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setXYZ(start_x, start_y, start_z);
    start->as<ob::SO3StateSpace::StateType>(1)->setIdentity();

    // create a goal state
    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setXYZ(goal_x, goal_y, goal_z - 0.2);   //treshold for landing height 0.2
    goal->as<ob::SO3StateSpace::StateType>(1)->setIdentity();

    // create a problem instance
    ob::ProblemDefinitionPtr pdef(new ob::ProblemDefinition(si));

    // check the validity of chosen states
    ROS_INFO_STREAM("Vailidity of start point: " << isWaypointValid(start));
    ROS_INFO_STREAM("Vailidity of goal point: " << isWaypointValid(goal));

    // set the start and goal states
    pdef->setStartAndGoalStates(start, goal);

    // create a planner for the defined space
    ob::PlannerPtr planner(new og::RRTstar(si));

    // set the problem we are trying to solve for the planner
    planner->setProblemDefinition(pdef);

    // perform setup steps for the planner
    planner->setup();

    // print the settings for this space
    si->printSettings(std::cout);

    // print the problem settings
    pdef->print(std::cout);

    // attempt to solve the problem within one second of planning time
    ob::PlannerStatus solved = planner->solve(max_planning_duration);

    if (solved && isWaypointValid(start) && isWaypointValid(goal) && !pdef->hasApproximateSolution())
    {
        // get the goal representation from the problem definition (not the same as the goal state)
        // and inquire about the found path
        ROS_INFO("Found solution:");
        ob::PathPtr path = pdef->getSolutionPath();
        og::PathGeometric* pth = pdef->getSolutionPath()->as<og::PathGeometric>();

        // print the path to screen
        pth->printAsMatrix(std::cout);
        trajectory_msgs::MultiDOFJointTrajectory msg;
        trajectory_msgs::MultiDOFJointTrajectoryPoint point_msg;

        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "world";
        msg.joint_names.clear();
        msg.points.clear();
        msg.joint_names.push_back("quadcopter");

        for (std::size_t path_idx = 0; path_idx < pth->getStateCount (); path_idx++)
        {
            const ob::SE3StateSpace::StateType *se3state = pth->getState(path_idx)->as<ob::SE3StateSpace::StateType>();

            // extract the first component of the state and cast it to what we expect
            const ob::RealVectorStateSpace::StateType *pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);

            // extract the second component of the state and cast it to what we expect
            const ob::SO3StateSpace::StateType *rot = se3state->as<ob::SO3StateSpace::StateType>(1);

            point_msg.time_from_start.fromSec(ros::Time::now().toSec());
            point_msg.transforms.resize(1);

            point_msg.transforms[0].translation.x= pos->values[0];
            point_msg.transforms[0].translation.y = pos->values[1];
            point_msg.transforms[0].translation.z = pos->values[2];

            point_msg.transforms[0].rotation.x = rot->x;
            point_msg.transforms[0].rotation.y = rot->y;
            point_msg.transforms[0].rotation.z = rot->z;
            point_msg.transforms[0].rotation.w = rot->w;

            msg.points.push_back(point_msg);
        }

        //call mav_trajectory now
        waypointsCallback(msg);
    }
    else
    {
      ROS_WARN("No solution found");
    }
}

void cust_callback(const octomap_msgs::Octomap::ConstPtr &msg, const geometry_msgs::PoseStamped::ConstPtr &pose)
{
    // convert octree to collision object
    octomap::OcTree* tree_oct = dynamic_cast<octomap::OcTree*>(octomap_msgs::msgToMap(*msg));
    fcl::OcTree* tree = new fcl::OcTree(std::shared_ptr<const octomap::OcTree>(tree_oct));
    fcl::CollisionObject temp((std::shared_ptr<fcl::CollisionGeometry>(tree)));
    treeObj = temp;

    ROS_DEBUG("Octomap loaded!");

    start_x = pose->pose.position.x;
    start_y = pose->pose.position.y;
    start_z = pose->pose.position.z;

    flag_sub_goal = 1;

    ROS_DEBUG("Current starting pose loaded: (%f, %f, %f)", start_x, start_y, start_z);
}

void get_landing_point_callback(const geometry_msgs::Pose &pose)
{
    goal_x = pose.position.x;
    goal_y = pose.position.y;
    goal_z = pose.position.z;

    ROS_INFO("Landing pose loaded: (%f, %f, %f)", goal_x, goal_y, goal_z);

    if(flag_sub_goal == 1)
    {
      // create markers for starting and ending point_msg
      visualization_msgs::Marker start_marker_msg = get_extreme_points_marker_from_position(start_x, start_y, start_z, ros::Time(), 0);
      visualization_msgs::Marker goal_marker_msg = get_extreme_points_marker_from_position(goal_x, goal_y, goal_z, ros::Time(), 1);
      // publish extreme points
      start_pub.publish(start_marker_msg);
      goal_pub.publish(goal_marker_msg);
      // plan trajectory
      plan_path_to_goal();
    }
}

int main(int argc, char **argv)
{
    // init ROS......
    ros::init(argc, argv, "eth_mav_octomap_planner");
    ros::NodeHandle nh("~");

    // Read paramters.....
    nh.param<double>("max_planning_duration", max_planning_duration, 5.0);
    nh.param<bool>("is_bounding_box_flag", is_bounding_box_flag, true);

    // ROS Publishers.....
    vis_pub = nh.advertise<visualization_msgs::MarkerArray>( "/trajectory_landing/spline_marker_array" ,1 , true);
    start_pub = nh.advertise<visualization_msgs::Marker>( "/trajectory_landing/start_pose_marker", 1, true);
    goal_pub = nh.advertise<visualization_msgs::Marker>( "/trajectory_landing/goal_pose_marker", 1, true);

    // ROS SUbscribers.....
    using namespace message_filters;
    message_filters::Subscriber<octomap_msgs::Octomap> oct_sub(nh, "/octomap_binary", 1);
    message_filters::Subscriber<geometry_msgs::PoseStamped> cur_pose_sub(nh, "/airsim/pose", 1);

    typedef sync_policies::ApproximateTime<octomap_msgs::Octomap, geometry_msgs::PoseStamped> RetrieveSimDataPolicy;
    Synchronizer<RetrieveSimDataPolicy> sync(RetrieveSimDataPolicy(10000), oct_sub, cur_pose_sub);
    sync.registerCallback(boost::bind(&cust_callback, _1, _2));

    ros::Subscriber landing_pose_sub = nh.subscribe("/trajectory_landing/clicked_goal_pose", 1, get_landing_point_callback);

    ROS_INFO("OMPL version: %s \n", OMPL_VERSION);

    ros::spin();

    return 0;
}
