#include <ros/ros.h>
#include <ros/package.h>

#include <mav_visualization/helpers.h>
#include <mav_trajectory_generation/polynomial_optimization_linear.h>
#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <mav_trajectory_generation/timing.h>
#include "mav_trajectory_generation/trajectory_sampling.h"

#include "mav_trajectory_generation_ros/ros_conversions.h"
#include "mav_trajectory_generation_ros/ros_visualization.h"

namespace mav_trajectory_generation {

// Benchmarking utilities to evaluate different methods of time allocation for
// polynomial trajectories.

struct TimeAllocationBenchmarkResult {
  TimeAllocationBenchmarkResult()
      : trial_number(-1),
        method_name("none"),
        num_segments(0),
        nominal_length(0.0),
        optimization_success(false),
        bounds_violated(false),
        trajectory_time(0.0),
        trajectory_length(0.0),
        computation_time(0.0),
        v_min_actual(Extremum(0.0, 0.0, 0)),
        a_min_actual(Extremum(0.0, 0.0, 0)),
        v_max_actual(Extremum(0.0, 0.0, 0)),
        a_max_actual(Extremum(0.0, 0.0, 0)),
        abs_violation_v(0.0),
        abs_violation_a(0.0),
        rel_violation_v(0.0),
        rel_violation_a(0.0),
        max_dist_from_straight_line(0.0),
        area_traj_straight_line(0.0) {}

  // Evaluation settings
  int trial_number;
  std::string method_name;

  // Trajectory settings
  int num_segments;
  double nominal_length;

  // Evaluation results
  bool optimization_success;
  bool bounds_violated;
  double trajectory_time;
  double trajectory_length;
  double computation_time;
  Extremum v_min_actual;
  Extremum a_min_actual;
  Extremum v_max_actual;
  Extremum a_max_actual;
  double abs_violation_v;
  double abs_violation_a;
  double rel_violation_v;
  double rel_violation_a;
  double max_dist_from_straight_line;
  double area_traj_straight_line;

  // More to come: convex hull/bounding box, etc.
};

class TimeEvaluationNode {
 public:
  TimeEvaluationNode(const ros::NodeHandle& nh,
                     const ros::NodeHandle& nh_private);

  // Number of Coefficients
  const static int kN = 10;  // has to be even !!
  // Dimension
  const static int kDim = 3;

  // Running the actual benchmark, one trial at a time (so that it can be
  // paused between for visualization).
  void runBenchmark(int trial_number, int num_segments);

  // Generate trajectories with different methods.
  void runNfabian(const Vertex::Vector& vertices, Trajectory* trajectory) const;
  void runTrapezoidalTime(const Vertex::Vector& vertices,
                          Trajectory* trajectory) const;
  void runNonlinear(const Vertex::Vector& vertices,
                    Trajectory* trajectory) const;
  void runNonlinearRichter(const Vertex::Vector& vertices,
                           Trajectory* trajectory) const;
  void runMellingerOuterLoop(const Vertex::Vector& vertices,
                             bool use_trapezoidal_time,
                             Trajectory* trajectory) const;
  void runSegmentViolationScalingTime(const Vertex::Vector& vertices,
                                      Trajectory* trajectory) const;

  void evaluateTrajectory(const std::string& method_name,
                          const Trajectory& traj,
                          TimeAllocationBenchmarkResult* result) const;

  void visualizeTrajectory(const std::string& method_name,
                           const Trajectory& traj,
                           visualization_msgs::MarkerArray* markers);

  // Accessors.
  bool visualize() const { return visualize_; }

  // Helpers.
  visualization_msgs::Marker createMarkerForPath(
      mav_msgs::EigenTrajectoryPointVector& path,
      const std_msgs::ColorRGBA& color, const std::string& name,
      double scale = 0.05) const;

  bool computeMinMaxMagnitudeAllSegments(
          const Segment::Vector& segments, int derivative,
          const std::vector<int>& dimensions,
          std::vector<Extremum>* maxima) const;
  double computePathLength(mav_msgs::EigenTrajectoryPointVector& path) const;
  double computePointLineDistance(const Eigen::Vector3d& A,
                                  const Eigen::Vector3d& B,
                                  const Eigen::Vector3d& C) const;
  std::string printResults() const;
  void outputResults(
          const std::string& filename,
          const std::vector<TimeAllocationBenchmarkResult>& results) const;

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;

  // General settings.
  std::string frame_id_;
  bool visualize_;
  bool print_debug_info_;

  // Dynamic constraints.
  double v_max_;
  double a_max_;

  // General trajectory settings.
  int max_derivative_order_;

  // Store all the results.
  std::vector<TimeAllocationBenchmarkResult> results_;

  // ROS stuff.
  ros::Publisher path_marker_pub_;
};

TimeEvaluationNode::TimeEvaluationNode(const ros::NodeHandle& nh,
                                       const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      frame_id_("world"),
      visualize_(true),
      print_debug_info_(false),
      v_max_(1.0),
      a_max_(2.0),
      max_derivative_order_(derivative_order::JERK) {
  nh_private_.param("frame_id", frame_id_, frame_id_);
  nh_private_.param("visualize", visualize_, visualize_);
  nh_private_.param("v_max", v_max_, v_max_);
  nh_private_.param("a_max", a_max_, a_max_);

  path_marker_pub_ =
      nh_private_.advertise<visualization_msgs::MarkerArray>("path", 1, true);
}

void TimeEvaluationNode::runBenchmark(int trial_number, int num_segments) {
  srand(trial_number);

  const Eigen::VectorXd min_pos = Eigen::VectorXd::Constant(kDim, -5.0);
  const Eigen::VectorXd max_pos = -min_pos;

  // Use trial number as seed to create the trajectory.
  Vertex::Vector vertices;
  const double side_length = 10.0;
  const int rounds = 2;
  vertices = createSquareVertices(max_derivative_order_,
                                  Eigen::Vector3d::Zero(), side_length, rounds);

  TimeAllocationBenchmarkResult result;
  // Fill in all the basics in the results that are shared between all the
  // evaluations.
  result.trial_number = trial_number;
  result.num_segments = num_segments;

  // Compute nominal length from the vertices
  double nominal_length = 0.0;
  for (size_t i = 0; i < vertices.size()-1; ++i) {
    Eigen::VectorXd start, end;
    vertices[i].getConstraint(derivative_order::POSITION, &start);
    // Find first vertex with position constraint.
    size_t end_idx = i + 1;
    for (size_t j = end_idx; j < vertices.size(); ++j) {
      if (vertices[j].getConstraint(derivative_order::POSITION, &end)) {
        end_idx = j;
        break;
      }
    }
    const double segment_length = (end.head(3)-start.head(3)).norm();
    nominal_length += segment_length;
  }
  result.nominal_length = nominal_length;

  visualization_msgs::MarkerArray markers;

  // Run all the evaluations.
  std::string method_name = "nfabian";
  Trajectory trajectory_nfabian;
  timing::Timer timer_nfabian(method_name);
  runNfabian(vertices, &trajectory_nfabian);
  timer_nfabian.Stop();
  evaluateTrajectory(method_name, trajectory_nfabian, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name, trajectory_nfabian, &markers);
  }

  method_name = "trapezoidal";
  Trajectory trajectory_trapezoidal;
  timing::Timer timer_trapezoidal(method_name);
  runTrapezoidalTime(vertices, &trajectory_trapezoidal);
  timer_trapezoidal.Stop();
  evaluateTrajectory(method_name, trajectory_trapezoidal, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name, trajectory_trapezoidal, &markers);
  }

  method_name = "nonlinear";
  Trajectory trajectory_nonlinear;
  timing::Timer timer_nonlinear(method_name);
  runNonlinear(vertices, &trajectory_nonlinear);
  timer_nonlinear.Stop();
  evaluateTrajectory(method_name, trajectory_nonlinear, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name, trajectory_nonlinear, &markers);
  }

  method_name = "nonlinear_richter";
  Trajectory trajectory_nonlinear_richter;
  timing::Timer timer_nonlinear_richter(method_name);
  runNonlinearRichter(vertices, &trajectory_nonlinear_richter);
  timer_nonlinear_richter.Stop();
  evaluateTrajectory(method_name, trajectory_nonlinear_richter, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name, trajectory_nonlinear_richter, &markers);
  }

  method_name = "mellinger_outer_loop";
  Trajectory trajectory_mellinger_outer_loop;
  timing::Timer timer_mellinger(method_name);
  runMellingerOuterLoop(vertices, false, &trajectory_mellinger_outer_loop);
  timer_mellinger.Stop();
  evaluateTrajectory(method_name, trajectory_mellinger_outer_loop, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name, trajectory_mellinger_outer_loop, &markers);
  }

  method_name = "mellinger_outer_loop_trapezoidal_init";
  Trajectory trajectory_mellinger_outer_loop_trapezoidal_init;
  timing::Timer timer_mellinger_trapezoidal(method_name);
  runMellingerOuterLoop(vertices, true, &trajectory_mellinger_outer_loop_trapezoidal_init);
  timer_mellinger_trapezoidal.Stop();
  evaluateTrajectory(method_name,
                     trajectory_mellinger_outer_loop_trapezoidal_init, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name,
                        trajectory_mellinger_outer_loop_trapezoidal_init,
                        &markers);
  }

  method_name = "segment_violation_scaling";
  Trajectory trajectory_segment_violation_scaling;
  timing::Timer timer_segment_violation_scaling(method_name);
  runSegmentViolationScalingTime(vertices,
                                 &trajectory_segment_violation_scaling);
  timer_segment_violation_scaling.Stop();
  evaluateTrajectory(method_name, trajectory_segment_violation_scaling, &result);
  results_.push_back(result);
  if (visualize_) {
    visualizeTrajectory(method_name, trajectory_segment_violation_scaling,
                        &markers);
  }

  if (visualize_) {
    path_marker_pub_.publish(markers);
  }
}

void TimeEvaluationNode::runNfabian(const Vertex::Vector& vertices,
                                    Trajectory* trajectory) const {
  std::vector<double> segment_times;
  segment_times =
      mav_trajectory_generation::estimateSegmentTimes(vertices, v_max_, a_max_);

  mav_trajectory_generation::PolynomialOptimization<kN> linopt(kDim);
  linopt.setupFromVertices(vertices, segment_times, max_derivative_order_);
  linopt.solveLinear();
  linopt.getTrajectory(trajectory);
}

void TimeEvaluationNode::runTrapezoidalTime(const Vertex::Vector& vertices,
                                            Trajectory* trajectory) const {
  std::vector<double> segment_times;
  const double kTimeFactor = 1.0;
  CHECK(mav_trajectory_generation::estimateSegmentTimesVelocityRamp(
      vertices, v_max_, a_max_, kTimeFactor, &segment_times));

  mav_trajectory_generation::PolynomialOptimization<kN> linopt(kDim);
  linopt.setupFromVertices(vertices, segment_times, max_derivative_order_);
  linopt.solveLinear();
  linopt.getTrajectory(trajectory);
}

void TimeEvaluationNode::runNonlinear(const Vertex::Vector& vertices,
                                      Trajectory* trajectory) const {
  std::vector<double> segment_times;
  segment_times =
      mav_trajectory_generation::estimateSegmentTimes(vertices, v_max_, a_max_);

  mav_trajectory_generation::NonlinearOptimizationParameters nlopt_parameters;
  nlopt_parameters.time_alloc_method ==
          NonlinearOptimizationParameters::kSquaredTimeAndConstraints;
  nlopt_parameters.print_debug_info_time_allocation = print_debug_info_;
  mav_trajectory_generation::PolynomialOptimizationNonLinear<kN> nlopt(
      kDim, nlopt_parameters);
  nlopt.setupFromVertices(vertices, segment_times, max_derivative_order_);
  nlopt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max_);
  nlopt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max_);
  nlopt.optimize();
  nlopt.getTrajectory(trajectory);
}

void TimeEvaluationNode::runNonlinearRichter(
        const Vertex::Vector& vertices,
        Trajectory* trajectory) const {
  std::vector<double> segment_times;
  segment_times =
      mav_trajectory_generation::estimateSegmentTimes(vertices, v_max_, a_max_);

  mav_trajectory_generation::NonlinearOptimizationParameters nlopt_parameters;
  nlopt_parameters.time_alloc_method =
          NonlinearOptimizationParameters::kRichterTimeAndConstraints;
  nlopt_parameters.print_debug_info_time_allocation = print_debug_info_;
  mav_trajectory_generation::PolynomialOptimizationNonLinear<kN> nlopt(
      kDim, nlopt_parameters);
  nlopt.setupFromVertices(vertices, segment_times, max_derivative_order_);
  nlopt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max_);
  nlopt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max_);
  nlopt.optimize();
  nlopt.getTrajectory(trajectory);
}

void TimeEvaluationNode::runMellingerOuterLoop(
        const Vertex::Vector& vertices, bool use_trapezoidal_time,
        Trajectory* trajectory) const {
  std::vector<double> segment_times;
  if (use_trapezoidal_time) {
    const double kTimeFactor = 1.0;
    CHECK(estimateSegmentTimesVelocityRamp(
            vertices, v_max_, a_max_, kTimeFactor, &segment_times));
  } else {
    segment_times = estimateSegmentTimes(vertices, v_max_, a_max_);
  }

  mav_trajectory_generation::NonlinearOptimizationParameters nlopt_parameters;
  nlopt_parameters.algorithm = nlopt::LD_LBFGS;
  nlopt_parameters.time_alloc_method =
          NonlinearOptimizationParameters::kMellingerOuterLoop;
  nlopt_parameters.print_debug_info_time_allocation = print_debug_info_;
  mav_trajectory_generation::PolynomialOptimizationNonLinear<kN> nlopt(
      kDim, nlopt_parameters);
  nlopt.setupFromVertices(vertices, segment_times, max_derivative_order_);
  nlopt.addMaximumMagnitudeConstraint(derivative_order::VELOCITY, v_max_);
  nlopt.addMaximumMagnitudeConstraint(derivative_order::ACCELERATION, a_max_);
  nlopt.optimize();
  nlopt.getTrajectory(trajectory);
}

void TimeEvaluationNode::runSegmentViolationScalingTime(
        const Vertex::Vector& vertices, Trajectory* trajectory) const {
  std::vector<double> segment_times;
  segment_times =
          mav_trajectory_generation::estimateSegmentTimes(vertices, v_max_,
                                                          a_max_);
  mav_trajectory_generation::PolynomialOptimization<kN> linopt(kDim);
  linopt.setupFromVertices(vertices, segment_times, max_derivative_order_);
  linopt.solveLinear();
  linopt.getTrajectory(trajectory);

  // Check violation and rescale segments
  Segment::Vector segments;
  trajectory->getSegments(&segments);

  // Get relative violation at each segment
  // Taken and modified from Trajectory::computeMinMaxMagnitude()
  std::vector<int> dimensions = {0, 1, 2}; // Evaluate dimensions in x, y and z
  std::vector<Extremum> maxima_vel, maxima_acc;
  computeMinMaxMagnitudeAllSegments(segments, derivative_order::VELOCITY,
                                    dimensions, &maxima_vel);
  computeMinMaxMagnitudeAllSegments(segments, derivative_order::ACCELERATION,
                                    dimensions, &maxima_acc);

  // Print segment times before scaling
  if (print_debug_info_) {
    std::cout << "[Violation Scaling Original]: "
              << std::accumulate(segment_times.begin(), segment_times.end(),
                                 0.0) << std::endl;
  }
  // Scale segment times according to violation
  for (int i = 0; i < segment_times.size(); ++i) {

    // Evaluate constraint/bound violation
    double abs_violation_v, abs_violation_a, rel_violation_v, rel_violation_a;
    abs_violation_v = maxima_vel[i].value - v_max_;
    abs_violation_a = maxima_acc[i].value - a_max_;
    rel_violation_v = abs_violation_v / v_max_;
    rel_violation_a = abs_violation_a / a_max_;

    double smallest_rel_violation = std::max(rel_violation_a, rel_violation_v);

    if (print_debug_info_) {
      std::cout << i << " segment time: " << segment_times[i]
                << " | rel_vio_v: " << rel_violation_v
                << " | rel_vio_a: " << rel_violation_a << std::endl;
    }

    segment_times[i] /= (1.0-smallest_rel_violation);
  }

  // Check and make sure that segment times are > kOptimizationTimeLowerBound
  for (double& t : segment_times) {
    t = std::max(kOptimizationTimeLowerBound, t);
  }

  // Solve again with new segment times scaled according to relative violations
  linopt.updateSegmentTimes(segment_times);
  linopt.solveLinear();
  linopt.getTrajectory(trajectory);

  // Check violation and rescale segments
  Segment::Vector segments_after;
  trajectory->getSegments(&segments_after);

  // TODO: only debug
  // Check violation afterwards
  std::vector<Extremum> maxima_vel_after, maxima_acc_after;
  computeMinMaxMagnitudeAllSegments(segments_after, derivative_order::VELOCITY,
                                    dimensions, &maxima_vel_after);
  computeMinMaxMagnitudeAllSegments(segments_after,
                                    derivative_order::ACCELERATION,
                                    dimensions, &maxima_acc_after);

  // Print segment times after scaling
  if (print_debug_info_) {
    std::cout << "[Violation Scaling Solution]: "
              << std::accumulate(segment_times.begin(), segment_times.end(),
                                 0.0) << std::endl;
  }

  for (int m = 0; m < segments_after.size(); ++m) {
    double abs_violation_v, abs_violation_a, rel_violation_v, rel_violation_a;
    abs_violation_v = maxima_vel_after[m].value - v_max_;
    abs_violation_a = maxima_acc_after[m].value - a_max_;
    rel_violation_v = abs_violation_v / v_max_;
    rel_violation_a = abs_violation_a / a_max_;

    if (print_debug_info_) {
      std::cout << m << " segment time: " << segment_times[m]
                << " | rel_vio_v: " << rel_violation_v
                << " | rel_vio_a: " << rel_violation_a << std::endl;
    }
  }
}

void TimeEvaluationNode::visualizeTrajectory(
    const std::string& method_name, const Trajectory& traj,
    visualization_msgs::MarkerArray* markers) {
  // Maybe hash the method name to a color somehow????
  // Just hardcode it for now per method name.
  mav_visualization::Color trajectory_color;

  if (method_name == "nfabian") {
    trajectory_color = mav_visualization::Color::Yellow();
  } else if (method_name == "trapezoidal") {
    trajectory_color = mav_visualization::Color::Green();
  } else if (method_name == "nonlinear") {
    trajectory_color = mav_visualization::Color::Red();
  } else if (method_name == "nonlinear_richter") {
    trajectory_color = mav_visualization::Color::Blue();
  } else if (method_name == "mellinger_outer_loop") {
    trajectory_color = mav_visualization::Color::Orange();
  } else if (method_name == "mellinger_outer_loop_trapezoidal_init") {
    trajectory_color = mav_visualization::Color::Gray();
  } else if (method_name == "segment_violation_scaling") {
    trajectory_color = mav_visualization::Color::Chartreuse();
  } else {
    trajectory_color = mav_visualization::Color::White();
  }

  const double kDefaultSamplingTime = 0.1;  // In seconds.
  mav_msgs::EigenTrajectoryPointVector path;
  sampleWholeTrajectory(traj, kDefaultSamplingTime, &path);

  visualization_msgs::Marker marker;
  marker = createMarkerForPath(path, trajectory_color, method_name);

  markers->markers.push_back(marker);
}

void TimeEvaluationNode::evaluateTrajectory(
    const std::string& method_name, const Trajectory& traj,
    TimeAllocationBenchmarkResult* result) const {
  result->method_name = method_name;

  result->trajectory_time = traj.getMaxTime();

  // Evaluate path length.
  const double kDefaultSamplingTime = 0.1;  // In seconds.
  mav_msgs::EigenTrajectoryPointVector path;
  mav_trajectory_generation::sampleWholeTrajectory(traj, kDefaultSamplingTime,
                                                   &path);
  result->trajectory_length = computePathLength(path);
  result->computation_time = timing::Timing::GetTotalSeconds(method_name);

  // Evaluate min/max extrema
  std::vector<int> dimensions = {0, 1, 2}; // Evaluate dimensions in x, y and z
  traj.computeMinMaxMagnitude(derivative_order::VELOCITY, dimensions,
                              &result->v_min_actual, &result->v_max_actual);
  traj.computeMinMaxMagnitude(derivative_order::ACCELERATION, dimensions,
                              &result->a_min_actual, &result->a_max_actual);

  // Evaluate constraint/bound violation
  result->abs_violation_v = result->v_max_actual.value - v_max_;
  result->abs_violation_a = result->a_max_actual.value - a_max_;
  result->rel_violation_v = result->abs_violation_v / v_max_;
  result->rel_violation_a = result->abs_violation_a / a_max_;
  if ((result->abs_violation_a > 0.0) || (result->abs_violation_v > 0.0)) {
    result->bounds_violated = true;
  } else {
    result->bounds_violated = false;
  }

  // Todo: Add success variable to check for allowed relative violation, ...
  // const double allowed_rel_violation = 0.1;

  // Evaluate maximum trajectory distance per segment from straight line path
  // 1) Sample trajectory
  // 2) Check for biggest distance in each segment
  std::vector<Segment> segments;
  traj.getSegments(&segments);

  double max_dist = 0.0;
  double prev_dist = 0.0;
  double dist = 0.0;
  double area = 0.0;
  Eigen::Vector3d prev_pos, point;
  for (const auto& segment : segments) {
    // Get start and end of segment
    Eigen::Vector3d start = segment.evaluate(0.0, derivative_order::POSITION);
    Eigen::Vector3d end = segment.evaluate(segment.getTime(),
                                           derivative_order::POSITION);
    // Set point to start position of segment
    point = start;
    for (double t = 0.0; t < segment.getTime(); t+=kDefaultSamplingTime) {
      // Get previous and current position on trajectory
      prev_pos = point;
      point = segment.evaluate(t, derivative_order::POSITION);

      // Absolute distance of point AP from line BC
      prev_dist = dist;
      dist = computePointLineDistance(point, start, end);
      if (dist > max_dist) { max_dist = dist; }

      // Integrate area
      area += 0.5*(dist+prev_dist) * (point-prev_pos).norm();
    }
  }
  // TODO: Distinguish max_dist for each segment?
  result->max_dist_from_straight_line = max_dist;
  result->area_traj_straight_line = area;
}

bool TimeEvaluationNode::computeMinMaxMagnitudeAllSegments(
        const Segment::Vector& segments, int derivative,
        const std::vector<int>& dimensions, std::vector<Extremum>* maxima) const {
  // For all segments in the trajectory:
  for (size_t segment_idx = 0; segment_idx < segments.size(); segment_idx++) {
    // Compute candidates.
    std::vector<Extremum> candidates;
    if (!segments[segment_idx].computeMinMaxMagnitudeCandidates(
            derivative, 0.0, segments[segment_idx].getTime(), dimensions,
            &candidates)) {
      return false;
    }
    // Evaluate candidates.
    Extremum minimum_candidate, maximum_candidate;
    if (!segments[segment_idx].selectMinMaxMagnitudeFromCandidates(
            0.0, segments[segment_idx].getTime(), derivative, dimensions,
            candidates, &minimum_candidate, &maximum_candidate)) {
      return false;
    }
    maxima->push_back(maximum_candidate);
  }
  return true;
}

visualization_msgs::Marker TimeEvaluationNode::createMarkerForPath(
    mav_msgs::EigenTrajectoryPointVector& path,
    const std_msgs::ColorRGBA& color, const std::string& name,
    double scale) const {
  visualization_msgs::Marker path_marker;

  const int kPublishEveryNSamples = 1;
  const double kMaxMagnitude = 100.0;

  path_marker.header.frame_id = "world";

  path_marker.header.stamp = ros::Time::now();
  path_marker.type = visualization_msgs::Marker::LINE_STRIP;
  path_marker.color = color;
  path_marker.ns = name;
  path_marker.scale.x = scale;

  path_marker.points.reserve(path.size() / kPublishEveryNSamples);
  int i = 0;
  for (const mav_msgs::EigenTrajectoryPoint& point : path) {
    i++;
    if (i % kPublishEveryNSamples != 0) {
      continue;
    }
    // Check that we're in some reasonable bounds.
    // Makes rviz stop crashing.
    if (point.position_W.maxCoeff() > kMaxMagnitude ||
        point.position_W.minCoeff() < -kMaxMagnitude) {
      continue;
    }

    geometry_msgs::Point point_msg;
    tf::pointEigenToMsg(point.position_W, point_msg);
    path_marker.points.push_back(point_msg);
  }

  return path_marker;
}


double TimeEvaluationNode::computePointLineDistance(
        const Eigen::Vector3d& A, const Eigen::Vector3d& B,
        const Eigen::Vector3d& C) const {
  // Distance of point A from line CB
  Eigen::Vector3d d = (C - B) / (C-B).norm();
  Eigen::Vector3d v = A - B;
  double t = v.dot(d);
  Eigen::Vector3d P = B + t * d;
  return (P-A).norm();
}

double TimeEvaluationNode::computePathLength(
    mav_msgs::EigenTrajectoryPointVector& path) const {
  Eigen::Vector3d last_point;
  double distance = 0;
  for (int i = 0; i < path.size(); ++i) {
    const mav_msgs::EigenTrajectoryPoint& point = path[i];

    if (i > 0) {
      distance += (point.position_W - last_point).norm();
    }
    last_point = point.position_W;
  }

  return distance;
}

std::string TimeEvaluationNode::printResults() const {
  // TODO: add optimization_success
  std::stringstream s;
  // Header.
  s << "trial_number, method_name, num_segments, nominal_length, "
       "optimization_success, bounds_violated, trajectory_time, "
       "trajectory_length, computation_time, a_max_actual, v_max_actual, "
       "abs_violation_a, abs_violation_v, rel_violation_a, rel_violation_v, "
       "max_dist_sl_traj, area_traj_sl"
    << std::endl;
  for (size_t i = 0; i < results_.size(); ++i) {
    s << results_[i].trial_number << ", " << results_[i].method_name << ", "
      << results_[i].num_segments << ", " << results_[i].nominal_length << ", "
      << results_[i].optimization_success << ", " << results_[i].bounds_violated
      << ", " << results_[i].trajectory_time << ", "
      << results_[i].trajectory_length << ", " << results_[i].computation_time
      << ", " << results_[i].a_max_actual.value << ", "
      << results_[i].v_max_actual.value << ", " << results_[i].abs_violation_a
      << ", " << results_[i].abs_violation_v << ", "
      << results_[i].rel_violation_a << ", " << results_[i].rel_violation_v
      << ", " << results_[i].max_dist_from_straight_line
      << ", " << results_[i].area_traj_straight_line
      << std::endl;
  }

  // File path.
  std::string path = ros::package::getPath("mav_trajectory_generation_ros");
  std::string filename = "/results_time_allocation.csv";
  std::string results_path = path + filename;
  outputResults(results_path, results_);
  ROS_INFO("Save results at %s", results_path.c_str());

  return s.str();
}

void TimeEvaluationNode::outputResults(
        const std::string& filename,
        const std::vector<TimeAllocationBenchmarkResult>& results) const {
  // Append new lines to file
  FILE* fp = fopen(filename.c_str(), "w+");
  if (fp == NULL) {
    std::cout << "Cannot open file. ABORT wrinting results!" << std::endl;
    return;
  }

  fprintf(fp,
          "trial_num, method_name, #segments, nominal_length, "
                  "optimization_success, bounds_violated, trajectory_time, "
                  "trajectory_length, computation_time, a_max_actual,"
                  " v_max_actual, abs_violation_a, abs_violation_v, "
                  "rel_violation_a, rel_violation_v, max_dist_sl_traj,"
                  "area_traj_sl\n");
  for (const TimeAllocationBenchmarkResult& result : results) {
    fprintf(fp, "%d,%s,%d,%f,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
            result.trial_number, result.method_name.c_str(),
            result.num_segments, result.nominal_length,
            result.optimization_success, result.bounds_violated,
            result.trajectory_time, result.trajectory_length,
            result.computation_time, result.a_max_actual.value,
            result.v_max_actual.value, result.abs_violation_a,
            result.abs_violation_v, result.rel_violation_a,
            result.rel_violation_v, result.max_dist_from_straight_line,
            result.area_traj_straight_line);
  }

  fclose(fp);
}

}  // namespace mav_trajectory_generation

int main(int argc, char** argv) {
  ros::init(argc, argv, "time_evaluation_node");
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  ros::NodeHandle nh("");
  ros::NodeHandle nh_private("~");

  mav_trajectory_generation::TimeEvaluationNode time_eval_node(nh, nh_private);

  ROS_INFO("Initialized time evaluation node.");

  int num_trial_per_num_segments = 5;
  std::vector<int> num_segments_vector = {1, 2, 10, 50};

  int start_trial_number = 0;

  nh_private.param("start_trial_number", start_trial_number,
                   start_trial_number);

  int trial_number = 0;

  for (int i = 0; i < num_segments_vector.size(); ++i) {
    for (int j = 0; j < num_trial_per_num_segments; ++j) {
      if (trial_number < start_trial_number) {
        trial_number++;
        continue;
      }
      ROS_INFO("Trial number %d Num segments: %d", trial_number,
               num_segments_vector[i]);
      std::srand(trial_number);
      time_eval_node.runBenchmark(trial_number, num_segments_vector[i]);
      trial_number++;
      ros::spinOnce();
      if (time_eval_node.visualize()) {
        ros::Duration(2.0).sleep();
        ros::spinOnce();
      }
      if (!ros::ok()) {
        ROS_ERROR("Aborted early.");
        return 1;
      }
    }
  }

  ROS_INFO("Finished evaluations.");
  ROS_INFO("Results:\n%s", time_eval_node.printResults().c_str());
  // Print all timing results
  mav_trajectory_generation::timing::Timing::Print(std::cout);

  ros::spin();
  return 0;
}
