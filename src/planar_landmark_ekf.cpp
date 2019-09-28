#include <ros/ros.h>
#include <geometry_msgs/Quaternion.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
// #include <Eigen/Core>
// #include <Eigen/LU>
#include <std_msgs/Float64MultiArray.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "planar_landmark_ekf_slam/PlanarFeature.h"
#include "planar_landmark_ekf_slam/PlanarFeatureArray.h"

class PlanarLandmarkEKF{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_inipose;
		ros::Subscriber sub_bias;
		ros::Subscriber sub_imu;
		ros::Subscriber sub_odom;
		ros::Subscriber sub_features;
		/*publish*/
		tf::TransformBroadcaster tf_broadcaster;
		ros::Publisher pub_pose;
		ros::Publisher pub_pc_lm_global;	//visualize
		ros::Publisher pub_pc_lm_local;	//visualize
		ros::Publisher pub_markerarray;	//visualize
		ros::Publisher pub_posearray;	//visualize
		ros::Publisher pub_variance;	//visualize
		/*const*/
		const int size_robot_state = 6;	//X, Y, Z, R, P, Y (Global)
		const int size_lm_state = 3;	//x, y, z (Local)
		/*objects*/
		Eigen::VectorXd X;
		Eigen::MatrixXd P;
		sensor_msgs::Imu bias;
		pcl::PointCloud<pcl::PointXYZ>::Ptr observation {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr landmarks_local {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr landmarks_global {new pcl::PointCloud<pcl::PointXYZ>};
		planar_landmark_ekf_slam::PlanarFeatureArray list_obs;
		planar_landmark_ekf_slam::PlanarFeatureArray list_lm;
		planar_landmark_ekf_slam::PlanarFeatureArray list_erased_lm;
		geometry_msgs::PoseArray landmark_origins;
		/*flags*/
		bool inipose_is_available = false;
		bool bias_is_available = false;
		bool first_callback_imu = true;
		bool first_callback_odom = true;
		/*time*/
		ros::Time time_publish;
		ros::Time time_imu_now;
		ros::Time time_imu_last;
		ros::Time time_odom_now;
		ros::Time time_odom_last;
		/*visualization*/
		visualization_msgs::MarkerArray planes;
		/*parameters*/
		double threshold_corr_dist;
		double threshold_corr_position_diff;
		/*class*/
		class RemoveUnavailableLM{
			private:
				planar_landmark_ekf_slam::PlanarFeatureArray list_lm_;
				planar_landmark_ekf_slam::PlanarFeatureArray list_removed_lm_;
				planar_landmark_ekf_slam::PlanarFeatureArray list_left_lm_;
				Eigen::VectorXd X_;
				Eigen::MatrixXd P_;
				int size_robot_state_;
				int size_lm_state_;
			public:
				RemoveUnavailableLM(planar_landmark_ekf_slam::PlanarFeatureArray list_lm, const Eigen::VectorXd X, const Eigen::MatrixXd P, int size_robot_state, int size_lm_state);
				void Remove(planar_landmark_ekf_slam::PlanarFeatureArray& list_lm, Eigen::VectorXd& X, Eigen::MatrixXd& P);
				void Recover(planar_landmark_ekf_slam::PlanarFeatureArray& list_lm, Eigen::VectorXd& X, Eigen::MatrixXd& P);
				bool CheckNormalIsInward_(const Eigen::Vector3d& Ng);
		};
	public:
		PlanarLandmarkEKF();
		void CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg);
		void CallbackBias(const sensor_msgs::ImuConstPtr& msg);
		void CallbackIMU(const sensor_msgs::ImuConstPtr& msg);
		void PredictionIMU(sensor_msgs::Imu imu, double dt);
		void CallbackOdom(const nav_msgs::OdometryConstPtr& msg);
		void PredictionOdom(nav_msgs::Odometry odom, double dt);
		void CallbackFeatures(const planar_landmark_ekf_slam::PlanarFeatureArrayConstPtr &msg);
		void DataSyncBeforeAssoc(void);
		void DataAssociation(void);
		bool Judge(planar_landmark_ekf_slam::PlanarFeature lm, planar_landmark_ekf_slam::PlanarFeature obs);
		void MergeLM(int parent_id, int child_id);
		void UpdateFeatures(void);
		void UpdateLMInfo(int lm_id);
		bool Innovation(planar_landmark_ekf_slam::PlanarFeature lm, planar_landmark_ekf_slam::PlanarFeature obs, Eigen::Vector3d& Z, Eigen::VectorXd& H, Eigen::MatrixXd& jH, Eigen::VectorXd& Y, Eigen::MatrixXd& S);
		void DataSyncAfterAssoc(void);
		void PushBackMarkerPlanes(planar_landmark_ekf_slam::PlanarFeature lm);
		void EraseLM(int index);
		void UpdateComputation(const Eigen::VectorXd& Z, const Eigen::VectorXd& H, const Eigen::MatrixXd& jH, const Eigen::VectorXd& Diag_sigma);
		bool CheckNormalIsInward(const Eigen::Vector3d& Ng);
		Eigen::Vector3d PlaneGlobalToLocal(const Eigen::Vector3d& Ng);
		Eigen::Vector3d PlaneLocalToGlobal(const Eigen::Vector3d& Nl);
		Eigen::Vector3d PointLocalToGlobal(const Eigen::Vector3d& Pl);
		void Publication();
		geometry_msgs::PoseStamped StateVectorToPoseStamped(void);
		Eigen::Matrix3d GetRotationXYZMatrix(const Eigen::Vector3d& RPY, bool inverse);
		void VectorVStack(Eigen::VectorXd& A, const Eigen::VectorXd& B);
		void MatrixVStack(Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
		geometry_msgs::Quaternion QuatEigenToMsg(Eigen::Quaterniond q_eigen);
		double PiToPi(double angle);
};

PlanarLandmarkEKF::PlanarLandmarkEKF()
	:nhPrivate("~")
{
	/*subscribe*/
	sub_inipose = nh.subscribe("/initial_orientation", 1, &PlanarLandmarkEKF::CallbackInipose, this);
	sub_bias = nh.subscribe("/imu/bias", 1, &PlanarLandmarkEKF::CallbackBias, this);
	sub_imu = nh.subscribe("/imu/data", 1, &PlanarLandmarkEKF::CallbackIMU, this);
	sub_odom = nh.subscribe("/tinypower/odom", 1, &PlanarLandmarkEKF::CallbackOdom, this);
	sub_features = nh.subscribe("/features", 1, &PlanarLandmarkEKF::CallbackFeatures, this);
	/*publish*/
	pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/ekf/pose", 1);
	pub_posearray = nh.advertise<geometry_msgs::PoseArray>("/landmark_origins", 1);
	pub_pc_lm_global = nh.advertise<sensor_msgs::PointCloud2>("/landmark_global", 1);
	pub_pc_lm_local = nh.advertise<sensor_msgs::PointCloud2>("/landmark_local", 1);
	pub_markerarray = nh.advertise<visualization_msgs::MarkerArray>("planes", 1);
	pub_variance = nh.advertise<std_msgs::Float64MultiArray>("variance", 1);
	/*state*/
	X = Eigen::VectorXd::Zero(size_robot_state);
	const double initial_sigma = 0.001;
	P = initial_sigma*Eigen::MatrixXd::Identity(size_robot_state, size_robot_state);
	/*parameters*/
	nhPrivate.param("threshold_corr_dist", threshold_corr_dist, 0.1);
	std::cout << "threshold_corr_dist = " << threshold_corr_dist << std::endl;
	nhPrivate.param("threshold_corr_position_diff", threshold_corr_position_diff, 0.5);
	std::cout << "threshold_corr_position_diff = " << threshold_corr_position_diff << std::endl;
}

void PlanarLandmarkEKF::CallbackInipose(const geometry_msgs::QuaternionConstPtr& msg)
{
	if(!inipose_is_available){
		tf::Quaternion q_pose;
		quaternionMsgToTF(*msg, q_pose);
		tf::Matrix3x3(q_pose).getRPY(X(3), X(4), X(5));
		inipose_is_available = true;
		std::cout << "inipose_is_available = " << inipose_is_available << std::endl;
		std::cout << "initial robot state = " << std::endl << X.segment(0, size_robot_state) << std::endl;
	}
}

void PlanarLandmarkEKF::CallbackBias(const sensor_msgs::ImuConstPtr& msg)
{
	if(!bias_is_available){
		bias = *msg;
		bias_is_available = true;
	}
}

void PlanarLandmarkEKF::CallbackIMU(const sensor_msgs::ImuConstPtr& msg)
{
	/* std::cout << "Callback IMU" << std::endl; */

	time_publish = msg->header.stamp;
	time_imu_now = msg->header.stamp;
	double dt;
	try{
		dt = (time_imu_now - time_imu_last).toSec();
	}
	catch(std::runtime_error& ex) {
		ROS_ERROR("Exception: [%s]", ex.what());
	}
	time_imu_last = time_imu_now;
	if(first_callback_imu)	dt = 0.0;
	else if(inipose_is_available){
		/*angular velocity*/
		PredictionIMU(*msg, dt);
	}
	
	Publication();

	first_callback_imu = false;
}

void PlanarLandmarkEKF::PredictionIMU(sensor_msgs::Imu imu, double dt)
{
	/* std::cout << "Prediction IMU" << std::endl; */

	double x = X(0);
	double y = X(1);
	double z = X(2);
	double r_ = X(3);
	double p_ = X(4);
	double y_ = X(5);

	double delta_r = imu.angular_velocity.x*dt;
	double delta_p = imu.angular_velocity.y*dt;
	double delta_y = imu.angular_velocity.z*dt;
	if(bias_is_available){
		delta_r -= bias.angular_velocity.x*dt;
		delta_p -= bias.angular_velocity.y*dt;
		delta_y -= bias.angular_velocity.z*dt;
	}
	Eigen::Vector3d Drpy = {delta_r, delta_p, delta_y};
	
	int num_wall = (X.size() - size_robot_state)/size_lm_state;

	Eigen::Matrix3d Rot_rpy;	//normal rotation
	Rot_rpy <<	1,	sin(r_)*tan(p_),	cos(r_)*tan(p_),
				0,	cos(r_),			-sin(r_),
				0,	sin(r_)/cos(p_),	cos(r_)/cos(p_);

	/*F*/
	Eigen::VectorXd F(X.size());
	F.segment(0, 3) = X.segment(0, 3);
	F.segment(3, 3) = X.segment(3, 3) + Rot_rpy*Drpy;
	for(int i=3;i<6;i++)	F(i) = PiToPi(F(i));
	F.segment(size_robot_state, num_wall*size_lm_state) = X.segment(size_robot_state, num_wall*size_lm_state);

	/*jF*/
	Eigen::MatrixXd jF = Eigen::MatrixXd::Zero(X.size(), X.size());
	/*jF-xyz*/
	jF.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
	jF.block(0, 3, 3, 3) = Eigen::Matrix3d::Zero();
	jF.block(0, size_robot_state, 3, num_wall*size_lm_state) = Eigen::MatrixXd::Zero(3, num_wall*size_lm_state);
	/*jF-rpy*/
	jF.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
	jF(3, 3) = 1 + cos(r_)*tan(p_)*delta_p - sin(r_)*tan(p_)*delta_y;
	jF(3, 4) = sin(r_)/cos(p_)/cos(p_)*delta_p + cos(r_)/cos(p_)/cos(p_)*delta_y;
	jF(3, 5) = 0;
	jF(4, 3) = -sin(r_)*delta_p - cos(r_)*delta_y;
	jF(4, 4) = 1;
	jF(4, 5) = 0;
	jF(5, 3) = cos(r_)/cos(p_)*delta_p - sin(r_)/cos(p_)*delta_y;
	jF(5, 4) = sin(r_)*sin(p_)/cos(p_)/cos(p_)*delta_p + cos(r_)*sin(p_)/cos(p_)/cos(p_)*delta_y;
	jF(5, 5) = 1;
	jF.block(3, size_robot_state, 3, num_wall*size_lm_state) = Eigen::MatrixXd::Zero(3, num_wall*size_lm_state);
	/*jF-wall_xyz*/
	jF.block(size_robot_state, size_robot_state, num_wall*size_lm_state, num_wall*size_lm_state) = Eigen::MatrixXd::Identity(num_wall*size_lm_state, num_wall*size_lm_state);
	
	/*Q*/
	const double sigma = 1.0e-4;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	Q.block(0, 0, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
	Q.block(size_robot_state, size_robot_state, num_wall*size_lm_state, num_wall*size_lm_state) = Eigen::MatrixXd::Zero(num_wall*size_lm_state, num_wall*size_lm_state);
	
	/*Update*/
	X = F;
	P = jF*P*jF.transpose() + Q;
	
	/* std::cout << "X =" << std::endl << X << std::endl; */
	/* std::cout << "P =" << std::endl << P << std::endl; */
	/* std::cout << "jF =" << std::endl << jF << std::endl; */
}

void PlanarLandmarkEKF::CallbackOdom(const nav_msgs::OdometryConstPtr& msg)
{
	/* std::cout << "Callback Odom" << std::endl; */

	time_publish = msg->header.stamp;
	time_odom_now = msg->header.stamp;
	double dt;
	try{
		dt = (time_odom_now - time_odom_last).toSec();
	}
	catch(std::runtime_error& ex) {
		ROS_ERROR("Exception: [%s]", ex.what());
	}
	time_odom_last = time_odom_now;
	if(first_callback_odom)	dt = 0.0;
	else if(inipose_is_available)	PredictionOdom(*msg, dt);
	
	Publication();

	first_callback_odom = false;
}

void PlanarLandmarkEKF::PredictionOdom(nav_msgs::Odometry odom, double dt)
{
	/* std::cout << "Prediction Odom" << std::endl; */

	double x = X(0);
	double y = X(1);
	double z = X(2);
	double r_ = X(3);
	double p_ = X(4);
	double y_ = X(5);
	Eigen::Vector3d Dxyz = {odom.twist.twist.linear.x*dt, 0, 0};

	int num_wall = (X.size() - size_robot_state)/size_lm_state;

	/*F*/
	Eigen::VectorXd F(X.size());
	F.segment(0, 3) = X.segment(0, 3) + GetRotationXYZMatrix(X.segment(3, 3), false)*Dxyz;
	F.segment(3, 3) = X.segment(3, 3);
	F.segment(size_robot_state, num_wall*size_lm_state) = X.segment(size_robot_state, num_wall*size_lm_state);

	/*jF*/
	Eigen::MatrixXd jF = Eigen::MatrixXd::Zero(X.size(), X.size());
	/*jF-xyz*/
	jF.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity();
	jF(0, 3) = Dxyz(1)*(cos(r_)*sin(p_)*cos(y_) + sin(r_)*sin(y_)) + Dxyz(2)*(-sin(r_)*sin(p_)*cos(y_) + cos(r_)*sin(y_));
	jF(0, 4) = Dxyz(0)*(-sin(p_)*cos(y_)) + Dxyz(1)*(sin(r_)*cos(p_)*cos(y_)) + Dxyz(2)*(cos(r_)*cos(p_)*cos(y_));
	jF(0, 5) = Dxyz(0)*(-cos(p_)*sin(y_)) + Dxyz(1)*(-sin(r_)*sin(p_)*sin(y_) - cos(r_)*cos(y_)) + Dxyz(2)*(-cos(r_)*sin(p_)*sin(y_) + sin(r_)*cos(y_));
	jF(1, 3) = Dxyz(1)*(cos(r_)*sin(p_)*sin(y_) - sin(r_)*cos(y_)) + Dxyz(2)*(-sin(r_)*sin(p_)*sin(y_) - cos(r_)*cos(y_));
	jF(1, 4) = Dxyz(0)*(-sin(p_)*sin(y_)) + Dxyz(1)*(sin(r_)*cos(p_)*sin(y_)) + Dxyz(2)*(cos(r_)*cos(p_)*sin(y_));
	jF(1, 5) = Dxyz(0)*(cos(p_)*cos(y_)) + Dxyz(1)*(sin(r_)*sin(p_)*cos(y_) - cos(r_)*sin(y_)) + Dxyz(2)*(cos(r_)*sin(p_)*cos(y_) + sin(r_)*sin(y_));
	jF(2, 3) = Dxyz(1)*(cos(r_)*cos(p_)) + Dxyz(2)*(-sin(r_)*cos(p_)) ;
	jF(2, 4) = Dxyz(0)*(-cos(p_)) + Dxyz(1)*(-sin(r_)*sin(p_)) + Dxyz(2)*(-cos(r_)*sin(p_)) ;
	jF(2, 5) = 0;
	jF.block(0, size_robot_state, 3, num_wall*size_lm_state) = Eigen::MatrixXd::Zero(3, num_wall*size_lm_state);
	/*jF-rpy*/
	jF.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero();
	jF.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
	jF.block(3, size_robot_state, 3, num_wall*size_lm_state) = Eigen::MatrixXd::Zero(3, num_wall*size_lm_state);
	/*jF-wall_xyz*/
	jF.block(size_robot_state, size_robot_state, num_wall*size_lm_state, num_wall*size_lm_state) = Eigen::MatrixXd::Identity(num_wall*size_lm_state, num_wall*size_lm_state);

	/*Q*/
	const double sigma = 1.0e-4;
	Eigen::MatrixXd Q = sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	Q.block(3, 3, 3, 3) = Eigen::MatrixXd::Zero(3, 3);
	Q.block(size_robot_state, size_robot_state, num_wall*size_lm_state, num_wall*size_lm_state) = Eigen::MatrixXd::Zero(num_wall*size_lm_state, num_wall*size_lm_state);
	
	/* std::cout << "X =" << std::endl << X << std::endl; */
	/* std::cout << "P =" << std::endl << P << std::endl; */
	/* std::cout << "jF =" << std::endl << jF << std::endl; */
	/* std::cout << "F =" << std::endl << F << std::endl; */
	
	/*Update*/
	X = F;
	P = jF*P*jF.transpose() + Q;
}

void PlanarLandmarkEKF::CallbackFeatures(const planar_landmark_ekf_slam::PlanarFeatureArrayConstPtr &msg)
{
	std::cout << "===== Callback Features =====" << std::endl;
	std::cout << "msg->features.size() = " << msg->features.size() << std::endl;
	std::cout << "list_lm.features.size() = " << list_lm.features.size() << std::endl;

	double time_start = ros::Time::now().toSec();

	/*input*/
	time_publish = msg->header.stamp;
	list_obs = *msg;
	DataSyncBeforeAssoc();
	std::cout << "DataSyncBeforeAssoc point [s] = " << ros::Time::now().toSec() - time_start << std::endl;
	/*data association*/
	DataAssociation();
	std::cout << "DataAssociation point [s] = " << ros::Time::now().toSec() - time_start << std::endl;
	/*observation uodate*/
	UpdateFeatures();
	std::cout << "UpdateFeatures point [s] = " << ros::Time::now().toSec() - time_start << std::endl;
	/*Data Synchronization*/
	DataSyncAfterAssoc();
	std::cout << "DataSyncAfterAssoc point [s] = " << ros::Time::now().toSec() - time_start << std::endl;
	/*erase*/
	for(size_t i=0;i<list_lm.features.size();){
		if(list_lm.features[i].was_merged || list_lm.features[i].was_erased)	EraseLM(i);
		else	++i;
	}

	Publication();
	std::cout << "CallbackFeatures time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
}

void PlanarLandmarkEKF::DataSyncBeforeAssoc(void)
{
	/* std::cout << "Synchronization Wirh State Vector Before Association" << std::endl; */

	/*clear*/
	observation->points.clear();
	/*observation*/
	for(size_t i=0;i<list_obs.features.size();++i){
		/*input*/
		pcl::PointXYZ tmp_point;
		tmp_point.x = list_obs.features[i].point_local.x;
		tmp_point.y = list_obs.features[i].point_local.y;
		tmp_point.z = list_obs.features[i].point_local.z;
		observation->points.push_back(tmp_point);
		/*transformation*/
		Eigen::Vector3d Nl(
			list_obs.features[i].point_local.x,
			list_obs.features[i].point_local.y,
			list_obs.features[i].point_local.z
		);
		Eigen::Vector3d Ng = PlaneLocalToGlobal(Nl);
		std::vector<Eigen::Vector3d> list_minmax_point(4);
		list_minmax_point[0] = {
			list_obs.features[i].min_local.x,
			list_obs.features[i].min_local.y,
			list_obs.features[i].min_local.z
		};
		list_minmax_point[1] = {
			list_obs.features[i].max_local.x,
			list_obs.features[i].max_local.y,
			list_obs.features[i].max_local.z
		};
		list_minmax_point[2] = {
			list_obs.features[i].min_local.x,
			list_obs.features[i].max_local.y,
			list_obs.features[i].max_local.z
		};
		list_minmax_point[3] = {
			list_obs.features[i].max_local.x,
			list_obs.features[i].min_local.y,
			list_obs.features[i].min_local.z
		};
		Eigen::Vector3d MinGlobal, MaxGlobal;
		for(size_t j=0;j<list_minmax_point.size();j++){
			Eigen::Vector3d Tmp = PointLocalToGlobal(list_minmax_point[j]);
			if(j==0){
				MinGlobal = Tmp;
				MaxGlobal = Tmp;
			}
			else{
				for(size_t k=0;k<Tmp.size();k++){
					if(MinGlobal(k) > Tmp(k))	MinGlobal(k) = Tmp(k);
					if(MaxGlobal(k) < Tmp(k))	MaxGlobal(k) = Tmp(k);
				}
			}
		}
		Eigen::Vector3d Cent = MinGlobal + (MaxGlobal - MinGlobal)/2.0;
		/*input*/
		list_obs.features[i].min_global.x = MinGlobal(0);
		list_obs.features[i].min_global.y = MinGlobal(1);
		list_obs.features[i].min_global.z = MinGlobal(2);
		list_obs.features[i].max_global.x = MaxGlobal(0);
		list_obs.features[i].max_global.y = MaxGlobal(1);
		list_obs.features[i].max_global.z = MaxGlobal(2);
		list_obs.features[i].centroid.x = Cent(0);
		list_obs.features[i].centroid.y = Cent(1);
		list_obs.features[i].centroid.z = Cent(2);
		list_obs.features[i].point_global.x = Ng(0);
		list_obs.features[i].point_global.y = Ng(1);
		list_obs.features[i].point_global.z = Ng(2);
		list_obs.features[i].normal_is_inward = CheckNormalIsInward(Ng);
		list_obs.features[i].corr_id = -1;
		list_obs.features[i].was_observed_in_this_scan = true;
		list_obs.features[i].counter_match = 0;
		list_obs.features[i].counter_nomatch = 0;
		list_obs.features[i].was_merged = false;
		list_obs.features[i].was_erased = false;
	}
	/*landmarks*/
	for(int i=0;i<list_lm.features.size();++i){
		list_lm.features[i].was_observed_in_this_scan = false;
	}
}

void PlanarLandmarkEKF::DataAssociation(void)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(observation);
	/* const double search_radius = 0.1; */
	for(size_t i=0;i<list_lm.features.size();++i){
		/*kdtree search*/
		std::vector<int> neighbor_obs_id;
		std::vector<float> neighbor_obs_sqrdist;
		kdtree.radiusSearch(landmarks_local->points[i], threshold_corr_dist, neighbor_obs_id, neighbor_obs_sqrdist);
		for(size_t j=0;j<neighbor_obs_id.size();++j){
			std::cout << "lm_id = " << i << ", obs_id = " << neighbor_obs_id[j] << ": " << neighbor_obs_sqrdist[j] << std::endl;
		}
		if(neighbor_obs_id.size() == 0)	std::cout << "lm_id = " << i << ": No neighbor" << std::endl;
		/*search correspond*/
		for(size_t j=0;j<neighbor_obs_id.size();++j){
			bool flag_break = true;
			int obs_id = neighbor_obs_id[j];
			if(Judge(list_lm.features[i], list_obs.features[obs_id])){
				list_lm.features[i].corr_id = obs_id;
				list_lm.features[i].corr_dist = sqrt(neighbor_obs_sqrdist[j]);
				if(list_obs.features[obs_id].corr_id == -1)	list_obs.features[obs_id].corr_id = i;
				else{
					int tmp_corr_lm_id = list_obs.features[obs_id].corr_id;
					if(list_lm.features[tmp_corr_lm_id].list_lm_observed_simul[i]){
						/*compare*/
						if(list_lm.features[i].corr_dist < list_lm.features[tmp_corr_lm_id].corr_dist)	list_obs.features[obs_id].corr_id = i;
						else{
							list_obs.features[obs_id].corr_id = tmp_corr_lm_id;
							flag_break = false;
						}
					}
					else	MergeLM(tmp_corr_lm_id, i);
				}
				if(flag_break)	break;
			}
		}
	}
}

bool PlanarLandmarkEKF::Judge(planar_landmark_ekf_slam::PlanarFeature lm, planar_landmark_ekf_slam::PlanarFeature obs)
{
	/*judge in normal direction*/
	if(obs.normal_is_inward != lm.normal_is_inward)	return false;
	/*judge in position*/
	/* Eigen::Vector3d ObsMin( */
	/* 	obs.min_global.x, */
	/* 	obs.min_global.y, */
	/* 	obs.min_global.z */
	/* ); */
	/* Eigen::Vector3d ObsMax( */
	/* 	obs.max_global.x, */
	/* 	obs.max_global.y, */
	/* 	obs.max_global.z */
	/* ); */
	/* Eigen::Vector3d LmMin( */
	/* 	lm.min_global.x, */
	/* 	lm.min_global.y, */
	/* 	lm.min_global.z */
	/* ); */
	/* Eigen::Vector3d LmMax( */
	/* 	lm.max_global.x, */
	/* 	lm.max_global.y, */
	/* 	lm.max_global.z */
	/* ); */
	/* Eigen::Vector3d ObsCent = ObsMin + (ObsMax - ObsMin)/2.0; */
	/* Eigen::Vector3d LmCent = LmMin + (LmMax - LmMin)/2.0; */
	/* Eigen::Vector3d CentDist = (LmCent - ObsCent).cwiseAbs(); */
	/* Eigen::Vector3d SumWidth = (ObsMax - ObsMin).cwiseAbs()/2.0 + (LmMax - LmMin).cwiseAbs()/2.0; */

	Eigen::Vector3d ObsMin(
		obs.min_global.x,
		obs.min_global.y,
		obs.min_global.z
	);
	Eigen::Vector3d ObsMax(
		obs.max_global.x,
		obs.max_global.y,
		obs.max_global.z
	);
	Eigen::Vector3d LmMin(
		lm.min_global.x,
		lm.min_global.y,
		lm.min_global.z
	);
	Eigen::Vector3d LmMax(
		lm.max_global.x,
		lm.max_global.y,
		lm.max_global.z
	);
	Eigen::Vector3d ObsCent(
		obs.centroid.x,
		obs.centroid.y,
		obs.centroid.z
	);
	Eigen::Vector3d LmCent(
		lm.centroid.x,
		lm.centroid.y,
		lm.centroid.z
	);
	Eigen::Vector3d SumWidth = (ObsMax - ObsMin)/2.0 + (LmMax - LmMin)/2.0;
	Eigen::Vector3d CentDist = (LmCent - ObsCent).cwiseAbs();

	for(size_t i=0;i<CentDist.size();++i){
		/* if(CentDist(i) > SumWidth(i))	return false; */
		if(CentDist(i) > SumWidth(i) + threshold_corr_position_diff){
			std::cout << "ObsMin = (" << ObsMin(0) << ", " << ObsMin(1) << ", " << ObsMin(2) << ")" << std::endl;
			std::cout << "ObsMax = (" << ObsMax(0) << ", " << ObsMax(1) << ", " << ObsMax(2) << ")" << std::endl;
			std::cout << "LmMin = (" << LmMin(0) << ", " << LmMin(1) << ", " << LmMin(2) << ")" << std::endl;
			std::cout << "LmMax = (" << LmMax(0) << ", " << LmMax(1) << ", " << LmMax(2) << ")" << std::endl;
			return false;
		}
	}
	/*pass*/
	return true;
}

void PlanarLandmarkEKF::MergeLM(int parent_id, int child_id)
{
	/* std::cout << "Merge landmarks" << std::endl; */

	/*flag*/
	list_lm.features[child_id].was_merged = true;
	/*min-max*/
	if(list_lm.features[parent_id].min_global.x > list_lm.features[child_id].min_global.x)	list_lm.features[parent_id].min_global.x = list_lm.features[child_id].min_global.x;
	if(list_lm.features[parent_id].min_global.y > list_lm.features[child_id].min_global.y)	list_lm.features[parent_id].min_global.y = list_lm.features[child_id].min_global.y;
	if(list_lm.features[parent_id].min_global.z > list_lm.features[child_id].min_global.z)	list_lm.features[parent_id].min_global.z = list_lm.features[child_id].min_global.z;
	if(list_lm.features[parent_id].max_global.x < list_lm.features[child_id].max_global.x)	list_lm.features[parent_id].max_global.x = list_lm.features[child_id].max_global.x;
	if(list_lm.features[parent_id].max_global.y < list_lm.features[child_id].max_global.y)	list_lm.features[parent_id].max_global.y = list_lm.features[child_id].max_global.y;
	if(list_lm.features[parent_id].max_global.z < list_lm.features[child_id].max_global.z)	list_lm.features[parent_id].max_global.z = list_lm.features[child_id].max_global.z;
	/*list lm observed simul*/
	for(size_t i=0;i<list_lm.features[child_id].list_lm_observed_simul.size();++i){
		if(list_lm.features[child_id].list_lm_observed_simul[i]) list_lm.features[parent_id].list_lm_observed_simul[i] = list_lm.features[child_id].list_lm_observed_simul[i];
	}
	/*counter*/
	list_lm.features[parent_id].counter_match += list_lm.features[child_id].counter_match;
	list_lm.features[parent_id].counter_nomatch += list_lm.features[child_id].counter_nomatch;
}

void PlanarLandmarkEKF::UpdateFeatures(void)
{
	std::cout << "Update features" << std::endl;

	/*stack (new registration or update)*/
	Eigen::VectorXd Xnew(0);
	Eigen::VectorXd Zstacked(0);
	Eigen::VectorXd Hstacked(0);
	Eigen::MatrixXd jHstacked(0, 0);
	Eigen::VectorXd Diag_sigma(0);
	for(size_t i=0;i<list_obs.features.size();++i){
		if(list_obs.features[i].corr_id == -1){	//new landmark
			list_lm.features.push_back(list_obs.features[i]);
			/*stack*/
			Eigen::Vector3d Obs(
				list_obs.features[i].point_global.x,
				list_obs.features[i].point_global.y,
				list_obs.features[i].point_global.z
			);
			VectorVStack(Xnew, Obs);
		}
		else{	//associated observation
			int lm_id = list_obs.features[i].corr_id;
			/*update landmarks info*/
			UpdateLMInfo(lm_id);
			/*innovation*/
			Eigen::Vector3d Z;
			Eigen::VectorXd H;
			Eigen::MatrixXd jH;
			Eigen::VectorXd Y;
			Eigen::MatrixXd S;
			Innovation(list_lm.features[lm_id], list_obs.features[i], Z, H, jH, Y, S);
			/*stack*/
			VectorVStack(Zstacked, Z);
			VectorVStack(Hstacked, H);
			MatrixVStack(jHstacked, jH);
			double tmp_sigma = 0.1*1000/(double)list_obs.features[i].cluster_size;
			std::cout << "tmp_sigma = " << tmp_sigma << std::endl;
			VectorVStack(Diag_sigma, Eigen::Vector3d(tmp_sigma, tmp_sigma, tmp_sigma));
		}
	}
	/*update*/
	if(Zstacked.size()>0 && inipose_is_available)   UpdateComputation(Zstacked, Hstacked, jHstacked, Diag_sigma);
	/*new registration*/
	X.conservativeResize(X.size() + Xnew.size());
	X.segment(X.size() - Xnew.size(), Xnew.size()) = Xnew;
	Eigen::MatrixXd Ptmp = P;
	const double initial_lm_sigma = 0.001;
	P = initial_lm_sigma*Eigen::MatrixXd::Identity(X.size(), X.size());
	P.block(0, 0, Ptmp.rows(), Ptmp.cols()) = Ptmp;
}

void PlanarLandmarkEKF::UpdateLMInfo(int lm_id)
{
	/* std::cout << "Update landmark information" << std::endl; */

	list_lm.features[lm_id].was_observed_in_this_scan = true;
	int obs_id = list_lm.features[lm_id].corr_id;
	/*min-max*/
	if(list_lm.features[lm_id].min_global.x > list_obs.features[obs_id].min_global.x)	list_lm.features[lm_id].min_global.x = list_obs.features[obs_id].min_global.x;
	if(list_lm.features[lm_id].min_global.y > list_obs.features[obs_id].min_global.y)	list_lm.features[lm_id].min_global.y = list_obs.features[obs_id].min_global.y;
	if(list_lm.features[lm_id].min_global.z > list_obs.features[obs_id].min_global.z)	list_lm.features[lm_id].min_global.z = list_obs.features[obs_id].min_global.z;
	if(list_lm.features[lm_id].max_global.x < list_obs.features[obs_id].max_global.x)	list_lm.features[lm_id].max_global.x = list_obs.features[obs_id].max_global.x;
	if(list_lm.features[lm_id].max_global.y < list_obs.features[obs_id].max_global.y)	list_lm.features[lm_id].max_global.y = list_obs.features[obs_id].max_global.y;
	if(list_lm.features[lm_id].max_global.z < list_obs.features[obs_id].max_global.z)	list_lm.features[lm_id].max_global.z = list_obs.features[obs_id].max_global.z;
}

bool PlanarLandmarkEKF::Innovation(planar_landmark_ekf_slam::PlanarFeature lm, planar_landmark_ekf_slam::PlanarFeature obs, Eigen::Vector3d& Z, Eigen::VectorXd& H, Eigen::MatrixXd& jH, Eigen::VectorXd& Y, Eigen::MatrixXd& S)
{
	/* std::cout << "Innovation" << std::endl; */

	/*state*/
	Eigen::Vector3d Ng(
		lm.point_global.x,
		lm.point_global.y,
		lm.point_global.z
	);
	double d2 = Ng.dot(Ng);
	Eigen::Vector3d RPY = X.segment(3, 3);
	/*Z*/
	Z = {
		obs.point_local.x,
		obs.point_local.y,
		obs.point_local.z
	};
	/*H*/
	H = PlaneGlobalToLocal(Ng);
	/*jH*/
	jH = Eigen::MatrixXd::Zero(Z.size(), X.size());
	/*dH/d(XYZ)*/
	Eigen::Vector3d rotN = GetRotationXYZMatrix(RPY, true)*Ng;
	for(int j=0;j<Z.size();j++){
		for(int k=0;k<3;k++)	jH(j, k) = -Ng(k)/d2*rotN(j);
	}
	/*dH/d(RPY)*/
	Eigen::Vector3d delN = Ng - Ng.dot(X.segment(0, 3))/d2*Ng;
	jH(0, 3) = 0;
	jH(0, 4) = (-sin(RPY(1))*cos(RPY(2)))*delN(0) + (-sin(RPY(1))*sin(RPY(2)))*delN(1) + (-cos(RPY(1)))*delN(2);
	jH(0, 5) = (-cos(RPY(1))*sin(RPY(2)))*delN(0) + (cos(RPY(1))*cos(RPY(2)))*delN(1);
	jH(1, 3) = (cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)))*delN(0) + (cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)))*delN(1) + (cos(RPY(0))*cos(RPY(1)))*delN(2);
	jH(1, 4) = (sin(RPY(0))*cos(RPY(1))*cos(RPY(2)))*delN(0) + (sin(RPY(0))*cos(RPY(1))*sin(RPY(2)))*delN(1) + (-sin(RPY(0))*sin(RPY(1)))*delN(2);
	jH(1, 5) = (-sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)))*delN(0) + (sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)))*delN(1);
	jH(2, 3) = (-sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) + cos(RPY(0))*sin(RPY(2)))*delN(0) + (-sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) - cos(RPY(0))*cos(RPY(2)))*delN(1) + (-sin(RPY(0))*cos(RPY(1)))*delN(2);
	jH(2, 4) = (cos(RPY(0))*cos(RPY(1))*cos(RPY(2)))*delN(0) + (cos(RPY(0))*cos(RPY(1))*sin(RPY(2)))*delN(1) + (-cos(RPY(0))*sin(RPY(1)))*delN(2);
	jH(2, 5) = (-cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) + sin(RPY(0))*cos(RPY(2)))*delN(0) + (cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)))*delN(1);
	/*dH/d(LM)*/
	Eigen::Matrix3d Tmp;
	for(int j=0;j<Z.size();j++){
		for(int k=0;k<size_lm_state;k++){
			if(j==k)	Tmp(j, k) = 1 - ((Ng.dot(X.segment(0, 3)) + Ng(j)*X(k))/d2 - Ng(j)*Ng.dot(X.segment(0, 3))/(d2*d2)*2*Ng(k));
			else	Tmp(j, k) = -(Ng(j)*X(k)/d2 - Ng(j)*Ng.dot(X.segment(0, 3))/(d2*d2)*2*Ng(k));
		}
	}
	jH.block(0, size_robot_state + lm.id*size_lm_state, Z.size(), size_lm_state) = GetRotationXYZMatrix(RPY, true)*Tmp;
	/*R*/
	const double sigma = 1.0e-2;
	Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	/*Y, S*/
	Y = Z - H;
	S = jH*P*jH.transpose() + R;
}

void PlanarLandmarkEKF::DataSyncAfterAssoc(void)
{
	/* std::cout << "Synchronization Wirh State Vector After Association" << std::endl; */

	/*clear*/
	landmarks_local->points.clear();
	landmarks_global->points.clear();
	landmark_origins.poses.clear();
	planes.markers.clear();
	/*landmarks*/
	for(int i=0;i<list_lm.features.size();++i){
		/*transform*/
		Eigen::Vector3d Ng = X.segment(size_robot_state + i*size_lm_state, size_lm_state);
		Eigen::Vector3d Nl = PlaneGlobalToLocal(Ng);
		/*pc input*/
		pcl::PointXYZ tmp;
		tmp.x = Nl(0);
		tmp.y = Nl(1);
		tmp.z = Nl(2);
		landmarks_local->points.push_back(tmp);
		tmp.x = Ng(0);
		tmp.y = Ng(1);
		tmp.z = Ng(2);
		landmarks_global->points.push_back(tmp);
		/*input*/
		list_lm.features[i].point_global.x = Ng(0);
		list_lm.features[i].point_global.y = Ng(1);
		list_lm.features[i].point_global.z = Ng(2);
		list_lm.features[i].point_local.x = Nl(0);
		list_lm.features[i].point_local.y = Nl(1);
		list_lm.features[i].point_local.z = Nl(2);
		list_lm.features[i].centroid.x = list_lm.features[i].min_global.x + (list_lm.features[i].max_global.x - list_lm.features[i].min_global.x)/2.0;
		list_lm.features[i].centroid.y = list_lm.features[i].min_global.y + (list_lm.features[i].max_global.y - list_lm.features[i].min_global.y)/2.0;
		list_lm.features[i].centroid.z = list_lm.features[i].min_global.z + (list_lm.features[i].max_global.z - list_lm.features[i].min_global.z)/2.0;
		list_lm.features[i].id = i;
		list_lm.features[i].corr_id = -1;

		/*match or no-match*/
		const int threshold_counter_match = 100;
		const int threshold_counter_nomatch = 200;
		list_lm.features[i].list_lm_observed_simul.resize(list_lm.features.size(), false);	//keeps valuses and inputs "false" into new memories
		if(list_lm.features[i].was_observed_in_this_scan){
			/*counter*/
			list_lm.features[i].counter_match++;
			/*list lm observed simul*/
			for(size_t j=0;j<list_lm.features.size();++j){
				if(list_lm.features[j].was_observed_in_this_scan)	list_lm.features[i].list_lm_observed_simul[j] = true;
			}
		}
		else{
			/*counter*/
			list_lm.features[i].counter_nomatch++;
			if(list_lm.features[i].counter_match < threshold_counter_match && list_lm.features[i].counter_nomatch > threshold_counter_nomatch)	list_lm.features[i].was_erased = true;
		}

		/*visual origin position*/
		Eigen::Vector3d Cent(
			list_lm.features[i].centroid.x,
			list_lm.features[i].centroid.y,
			list_lm.features[i].centroid.z
		);
		Eigen::Vector3d NgToCent = Cent - Ng;
		Eigen::Vector3d Origin = Ng + (NgToCent - NgToCent.dot(Ng)/Ng.norm()/Ng.norm()*Ng);
		/*visual origin orientation*/
		std::vector<Eigen::Vector3d> tmp_axes(3);
		tmp_axes[0] = -Ng.normalized();
		if(!list_lm.features[i].normal_is_inward)	tmp_axes[0] *= -1;
		tmp_axes[1] = (Origin - Ng).normalized();
		tmp_axes[2] = (tmp_axes[0].cross(tmp_axes[1])).normalized();
		Eigen::Matrix3d Axes;
		for(size_t j=0;j<tmp_axes.size();++j)	Axes.block(0, j, 3, 1) = tmp_axes[j];
		Eigen::Quaterniond q_orientation(Axes);
		q_orientation.normalize();
		/*input*/
		list_lm.features[i].origin.position.x = Origin(0);
		list_lm.features[i].origin.position.y = Origin(1);
		list_lm.features[i].origin.position.z = Origin(2);
		// list_lm.features[i].origin.position.x = list_lm.features[i].centroid.x;
		// list_lm.features[i].origin.position.y = list_lm.features[i].centroid.y;
		// list_lm.features[i].origin.position.z = list_lm.features[i].centroid.z;
		list_lm.features[i].origin.orientation = QuatEigenToMsg(q_orientation);
		landmark_origins.poses.push_back(list_lm.features[i].origin);
		/*visual scale*/
		Eigen::Vector3d MinMax(
			list_lm.features[i].max_global.x - list_lm.features[i].min_global.x,
			list_lm.features[i].max_global.y - list_lm.features[i].min_global.y,
			list_lm.features[i].max_global.z - list_lm.features[i].min_global.z
		);
		list_lm.features[i].scale.x = (MinMax.dot(tmp_axes[0])/tmp_axes[0].norm()/tmp_axes[0].norm()*tmp_axes[0]).norm();
		list_lm.features[i].scale.y = (MinMax.dot(tmp_axes[1])/tmp_axes[1].norm()/tmp_axes[1].norm()*tmp_axes[1]).norm();
		list_lm.features[i].scale.z = (MinMax.dot(tmp_axes[2])/tmp_axes[2].norm()/tmp_axes[2].norm()*tmp_axes[2]).norm();
		/*input*/
		PushBackMarkerPlanes(list_lm.features[i]);
	}
	/*erased landmark*/
	for(int i=0;i<list_erased_lm.features.size();++i)	PushBackMarkerPlanes(list_erased_lm.features[i]);
}

void PlanarLandmarkEKF::EraseLM(int index)
{
	std::cout << "Erase landmark" << std::endl;

	/*keep*/
	list_erased_lm.features.push_back(list_lm.features[index]);
	/*list*/
	list_lm.features.erase(list_lm.features.begin() + index);
	/*delmit point*/
	int delimit0 = size_robot_state + index*size_lm_state;
	int delimit1 = size_robot_state + (index+1)*size_lm_state;
	/*X*/
	Eigen::VectorXd tmp_X = X;
	X.resize(X.size() - size_lm_state);
	X.segment(0, delimit0) = tmp_X.segment(0, delimit0);
	X.segment(delimit0, X.size() - delimit0) = tmp_X.segment(delimit1, tmp_X.size() - delimit1);
	/*P*/
	Eigen::MatrixXd tmp_P = P;
	P.resize(P.cols() - size_lm_state, P.rows() - size_lm_state);
	/*P-upper-left*/
	P.block(0, 0, delimit0, delimit0) = tmp_P.block(0, 0, delimit0, delimit0);
	/*P-upper-right*/
	P.block(0, delimit0, delimit0, P.cols()-delimit0) = tmp_P.block(0, delimit1, delimit0, tmp_P.cols()-delimit1);
	/*P-lower-left*/
	P.block(delimit0, 0, P.rows()-delimit0, delimit0) = tmp_P.block(delimit1, 0, tmp_P.rows()-delimit1, delimit0);
	/*P-lower-right*/
	P.block(delimit0, delimit0, P.rows()-delimit0, P.cols()-delimit0) = tmp_P.block(delimit1, delimit1, tmp_P.rows()-delimit1, tmp_P.cols()-delimit1);
	/*id*/
	for(size_t i=0;i<list_lm.features.size();++i){
		list_lm.features[i].id = i; 
		list_lm.features[i].list_lm_observed_simul.erase(list_lm.features[i].list_lm_observed_simul.begin() + index);
	}
}

bool PlanarLandmarkEKF::CheckNormalIsInward(const Eigen::Vector3d& Ng)
{
	Eigen::Vector3d VerticalPosition = X.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	double dot = VerticalPosition.dot(Ng);
	if(dot<0)	return true;
	else{
		double dist_wall = Ng.norm();
		double dist_robot = VerticalPosition.norm();
		if(dist_robot<dist_wall)	return true;
		else	return false;
	}
}

void PlanarLandmarkEKF::UpdateComputation(const Eigen::VectorXd& Z, const Eigen::VectorXd& H, const Eigen::MatrixXd& jH, const Eigen::VectorXd& Diag_sigma)
{
	std::cout << "Update computation" << std::endl;

	Eigen::VectorXd Y = Z - H;
	// const double sigma = 1.2e-1;	//using floor
	// Eigen::MatrixXd R = sigma*Eigen::MatrixXd::Identity(Z.size(), Z.size());
	Eigen::MatrixXd R = Diag_sigma.asDiagonal();
	Eigen::MatrixXd S = jH*P*jH.transpose() + R;
	Eigen::MatrixXd K = P*jH.transpose()*S.inverse();
	X = X + K*Y;
	for(int i=3;i<6;i++)	X(i) = PiToPi(X(i));
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(X.size(), X.size());
	P = (I - K*jH)*P;

	/* std::cout << "K = " << std::endl << K << std::endl; */
	/* std::cout << "K*Y = " << std::endl << K*Y << std::endl; */
}

Eigen::Vector3d PlanarLandmarkEKF::PlaneGlobalToLocal(const Eigen::Vector3d& Ng)
{
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	Eigen::Vector3d delL = Ng - DeltaVertical;
	Eigen::Vector3d Nl = GetRotationXYZMatrix(X.segment(3, 3), true)*delL;
	return Nl;
}

Eigen::Vector3d PlanarLandmarkEKF::PlaneLocalToGlobal(const Eigen::Vector3d& Nl)
{
	Eigen::Vector3d rotL = GetRotationXYZMatrix(X.segment(3, 3), false)*Nl;
	Eigen::Vector3d DeltaVertical = X.segment(0, 3).dot(rotL)/rotL.dot(rotL)*rotL;
	Eigen::Vector3d Ng = rotL + DeltaVertical;
	return Ng;
}

Eigen::Vector3d PlanarLandmarkEKF::PointLocalToGlobal(const Eigen::Vector3d& Pl)
{
	Eigen::Vector3d Pg = GetRotationXYZMatrix(X.segment(3, 3), false)*Pl + X.segment(0, 3);
	return Pg;
}

void PlanarLandmarkEKF::Publication(void)
{
	/* std::cout << "Publication" << std::endl; */

	for(int i=3;i<6;i++){	//test
		if(fabs(X(i))>M_PI){
			std::cout << "+PI -PI error" << std::endl;
			std::cout << "X(" << i << ") = " << X(i) << std::endl;
			exit(1);
		}
	}
	for(size_t i=0;i<X.size();i++){	//test
		if(std::isnan(X(i))){
			std::cout << "NAN error" << std::endl;
			std::cout << "X(" << i << ") = " << X(i) << std::endl;
			exit(1);
		}
	}

	/*pose*/
	geometry_msgs::PoseStamped pose_pub = StateVectorToPoseStamped();
	pose_pub.header.frame_id = "/odom";
	pose_pub.header.stamp = time_publish;
	pub_pose.publish(pose_pub);

	/*tf broadcast*/
    geometry_msgs::TransformStamped transform;
	transform.header.stamp = time_publish;
	transform.header.frame_id = "/odom";
	transform.child_frame_id = "/velodyne";
	transform.transform.translation.x = pose_pub.pose.position.x;
	transform.transform.translation.y = pose_pub.pose.position.y;
	transform.transform.translation.z = pose_pub.pose.position.z;
	transform.transform.rotation = pose_pub.pose.orientation;
	tf_broadcaster.sendTransform(transform);

	/*pc*/
	sensor_msgs::PointCloud2 pc_pub;
	pcl::toROSMsg(*landmarks_global, pc_pub);
	pc_pub.header.frame_id = "/odom";
	pc_pub.header.stamp = time_publish;
	pub_pc_lm_global.publish(pc_pub);

	/*landmark origins*/
	landmark_origins.header.frame_id = "/odom";
	landmark_origins.header.stamp = time_publish;
	pub_posearray.publish(landmark_origins);

	/*planes*/
	pub_markerarray.publish(planes);

	/*variance*/
	std_msgs::Float64MultiArray variance_pub;
	for(int i=0;i<P.cols();i++)	variance_pub.data.push_back(P(i, i));
	pub_variance.publish(variance_pub);

	/*pc*/
	sensor_msgs::PointCloud2 pc_lm_pred;
	pcl::toROSMsg(*landmarks_local, pc_lm_pred);
	pc_lm_pred.header.frame_id = "/velodyne";
	pc_lm_pred.header.stamp = time_publish;
	pub_pc_lm_local.publish(pc_lm_pred);
}

void PlanarLandmarkEKF::PushBackMarkerPlanes(planar_landmark_ekf_slam::PlanarFeature lm)
{
	const double thickness = 0.05;

	visualization_msgs::Marker tmp;
	tmp.header.frame_id = "/odom";
	tmp.header.stamp = time_imu_now;
	tmp.ns = "planes";
	tmp.id = planes.markers.size();
	tmp.action = visualization_msgs::Marker::ADD;
	tmp.pose = lm.origin;
	tmp.type = visualization_msgs::Marker::CUBE;
	tmp.scale = lm.scale;
	tmp.scale.x = thickness;
	if(lm.was_observed_in_this_scan){
		tmp.color.r = 1.0;
		tmp.color.g = 0.0;
		tmp.color.b = 0.0;
		tmp.color.a = 0.9;
	}
	else if(lm.was_merged){
		tmp.color.r = 1.0;
		tmp.color.g = 1.0;
		tmp.color.b = 0.0;
		tmp.color.a = 0.4;
	}
	else if(lm.was_erased){
		tmp.color.r = 1.0;
		tmp.color.g = 1.0;
		tmp.color.b = 1.0;
		tmp.color.a = 0.4;
	}
	else{
		tmp.color.r = 0.0;
		tmp.color.g = 0.0;
		tmp.color.b = 1.0;
		tmp.color.a = 0.9;
	}

	/*test*/
	const int highlighted_lm = 13;
	if(lm.id == highlighted_lm){
		tmp.color.r = 0.0;
		tmp.color.g = 1.0;
		tmp.color.b = 1.0;
		if(lm.was_observed_in_this_scan)	tmp.color.a = 0.9;
		else	tmp.color.a = 0.4;
	}
	/*test*/
	/* tmp.pose.orientation.x = 0; */
	/* tmp.pose.orientation.y = 0; */
	/* tmp.pose.orientation.z = 0; */
	/* tmp.pose.orientation.w = 1; */
	/* tmp.scale.x = lm.max_global.x - lm.min_global.x; */
	/* tmp.scale.y = lm.max_global.y - lm.min_global.y; */
	/* tmp.scale.z = lm.max_global.z - lm.min_global.z; */

	planes.markers.push_back(tmp);
}

geometry_msgs::PoseStamped PlanarLandmarkEKF::StateVectorToPoseStamped(void)
{
	geometry_msgs::PoseStamped pose;
	pose.pose.position.x = X(0);
	pose.pose.position.y = X(1);
	pose.pose.position.z = X(2);
	tf::Quaternion q_orientation = tf::createQuaternionFromRPY(X(3), X(4), X(5));
	pose.pose.orientation.x = q_orientation.x();
	pose.pose.orientation.y = q_orientation.y();
	pose.pose.orientation.z = q_orientation.z();
	pose.pose.orientation.w = q_orientation.w();

	return pose;
}

Eigen::Matrix3d PlanarLandmarkEKF::GetRotationXYZMatrix(const Eigen::Vector3d& RPY, bool inverse)
{
	Eigen::Matrix3d Rot_xyz;
	Rot_xyz <<
		cos(RPY(1))*cos(RPY(2)),	sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)),	cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)),
		cos(RPY(1))*sin(RPY(2)),	sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) + cos(RPY(0))*cos(RPY(2)),	cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)),
		-sin(RPY(1)),				sin(RPY(0))*cos(RPY(1)),										cos(RPY(0))*cos(RPY(1));
	
	Eigen::Matrix3d Rot_xyz_inv;
	Rot_xyz_inv <<
		cos(RPY(1))*cos(RPY(2)),										cos(RPY(1))*sin(RPY(2)),										-sin(RPY(1)),
		sin(RPY(0))*sin(RPY(1))*cos(RPY(2)) - cos(RPY(0))*sin(RPY(2)),	sin(RPY(0))*sin(RPY(1))*sin(RPY(2)) + cos(RPY(0))*cos(RPY(2)),	sin(RPY(0))*cos(RPY(1)),
		cos(RPY(0))*sin(RPY(1))*cos(RPY(2)) + sin(RPY(0))*sin(RPY(2)),	cos(RPY(0))*sin(RPY(1))*sin(RPY(2)) - sin(RPY(0))*cos(RPY(2)),	cos(RPY(0))*cos(RPY(1));

	if(!inverse)	return Rot_xyz;
	else	return Rot_xyz_inv;	//=Rot_xyz.transpose()
}

void PlanarLandmarkEKF::VectorVStack(Eigen::VectorXd& A, const Eigen::VectorXd& B)
{
	A.conservativeResize(A.rows() + B.rows());
	A.segment(A.size() - B.size(), B.size()) = B;
}

void PlanarLandmarkEKF::MatrixVStack(Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
	A.conservativeResize(A.rows() + B.rows(), B.cols());
	A.block(A.rows() - B.rows(), 0, B.rows(), B.cols()) = B;
}

geometry_msgs::Quaternion PlanarLandmarkEKF::QuatEigenToMsg(Eigen::Quaterniond q_eigen)
{
	geometry_msgs::Quaternion q_msg;
	q_msg.x = q_eigen.x();
	q_msg.y = q_eigen.y();
	q_msg.z = q_eigen.z();
	q_msg.w = q_eigen.w();
	return q_msg;
}

double PlanarLandmarkEKF::PiToPi(double angle)
{
	/* return fmod(angle + M_PI, 2*M_PI) - M_PI; */
	return atan2(sin(angle), cos(angle)); 
}

PlanarLandmarkEKF::RemoveUnavailableLM::RemoveUnavailableLM(planar_landmark_ekf_slam::PlanarFeatureArray list_lm, const Eigen::VectorXd X, const Eigen::MatrixXd P, int size_robot_state, int size_lm_state)
{
	list_lm_ = list_lm;
	X_ = X;
	P_ = P;
	size_robot_state_ = size_robot_state;
	size_lm_state_ = size_lm_state;
}
void PlanarLandmarkEKF::RemoveUnavailableLM::Remove(planar_landmark_ekf_slam::PlanarFeatureArray& list_lm, Eigen::VectorXd& X, Eigen::MatrixXd& P)
{
	const double max_observation_range = 10.0;
	for(size_t i=0;i<list_lm.features.size();){
		Eigen::Vector3d Ng(
			list_lm_.features[i].point_global.x,
			list_lm_.features[i].point_global.y,
			list_lm_.features[i].point_global.z
		);
		/*judge in direction of normal*/
		if(list_lm_.features[i].normal_is_inward == CheckNormalIsInward_(Ng)){
			i++;
			list_left_lm_.features.push_back(list_lm_.features[i]);
			continue;
		}
		/*judge in observation range*/
		Eigen::Vector3d LocalOrigin(
			list_lm_.features[i].centroid.x - X(0),
			list_lm_.features[i].centroid.y - X(1),
			list_lm_.features[i].centroid.z - X(2)
		);
		LocalOrigin = LocalOrigin.cwiseAbs();
		Eigen::Vector3d MinMax(
			(list_lm_.features[i].max_global.x - list_lm_.features[i].min_global.x)/2.0,
			(list_lm_.features[i].max_global.y - list_lm_.features[i].min_global.y)/2.0,
			(list_lm_.features[i].max_global.z - list_lm_.features[i].min_global.z)/2.0
		);
		if(LocalOrigin(0) < MinMax(0)+max_observation_range
			|| LocalOrigin(1) < MinMax(1)+max_observation_range
			|| LocalOrigin(2) < MinMax(2)+max_observation_range
		){
			i++;
			list_left_lm_.features.push_back(list_lm_.features[i]);
			continue;
		}
		/*remove*/
		int delimit0 = size_robot_state_ + i*size_lm_state_;
		int delimit1 = size_robot_state_ + (i+1)*size_lm_state_;
		/*X*/
		X.resize(X_.size() - size_lm_state_);
		X.segment(0, delimit0) = X_.segment(0, delimit0);
		X.segment(delimit0, X.size() - delimit0) = X_.segment(delimit1, X_.size() - delimit1);
		/*P*/
		P.resize(P_.cols() - size_lm_state_, P_.rows() - size_lm_state_);
		/*P-upper-left*/
		P.block(0, 0, delimit0, delimit0) = P_.block(0, 0, delimit0, delimit0);
		/*P-upper-right*/
		P.block(0, delimit0, delimit0, P.cols()-delimit0) = P_.block(0, delimit1, delimit0, P_.cols()-delimit1);
		/*P-lower-left*/
		P.block(delimit0, 0, P.rows()-delimit0, delimit0) = P_.block(delimit1, 0, P_.rows()-delimit1, delimit0);
		/*P-lower-right*/
		P.block(delimit0, delimit0, P.rows()-delimit0, P.cols()-delimit0) = P_.block(delimit1, delimit1, P_.rows()-delimit1, P_.cols()-delimit1);
		/*lm list*/
		list_removed_lm_.features.push_back(list_lm_.features[i]);
		list_lm.features.erase(list_lm.features.begin() + i);
		for(size_t j=0;j<list_lm.features.size();++j){
			list_lm.features[j].list_lm_observed_simul.erase(list_lm.features[j].list_lm_observed_simul.begin() + i);
			list_lm.features[j].id = j;
		}
	}
}
void PlanarLandmarkEKF::RemoveUnavailableLM::Recover(planar_landmark_ekf_slam::PlanarFeatureArray& list_lm, Eigen::VectorXd& X, Eigen::MatrixXd& P)
{
	/*robot state*/
	X_.segment(0, size_robot_state_) = X.segment(0, size_robot_state_);
	P_.block(0, 0, size_robot_state_, size_robot_state_) = P.block(0, 0, size_robot_state_, size_robot_state_);
	/*LM state*/
	for(size_t i=0;i<list_left_lm_.features.size();++i){
		X_.segment(size_robot_state_ + list_left_lm_.features[i].id*size_lm_state_, size_lm_state_) = X.segment(size_robot_state_ + i*size_lm_state_, size_lm_state_);
		/*P-row-left*/
		P_.block(
			size_robot_state_ + list_left_lm_.features[i].id*size_lm_state_,
			0,
			size_lm_state_,
			size_robot_state_
		) = P.block(size_robot_state_ + i*size_lm_state_, 0, size_lm_state_, size_robot_state_);
		/*P-col-upper*/
		P_.block(
			0,
			size_robot_state_ + list_left_lm_.features[i].id*size_lm_state_,
			size_robot_state_,
			size_lm_state_
		) = P.block(0, size_robot_state_ + i*size_lm_state_, size_robot_state_, size_lm_state_);
		/*p-inside*/
		for(size_t j=0;j<list_left_lm_.features.size();j++){
			P_.block(
				size_robot_state_ + list_left_lm_.features[i].id*size_lm_state_,
				size_robot_state_ + list_left_lm_.features[j].id*size_lm_state_,
				size_lm_state_,
				size_lm_state_
			) = P.block(size_robot_state_ + i*size_lm_state_, size_robot_state_ + j*size_lm_state_, size_lm_state_, size_lm_state_);
		}
	}
	X = X_;
	P = P_;
	/*lm list*/
	for(size_t i=0;i<list_removed_lm_.features.size();i++)	list_lm.features.insert(list_lm.features.begin() + list_removed_lm_.features[i].id, list_removed_lm_.features[i]);
	for(size_t i=0;i<list_lm.features.size();++i){
		list_lm.features[i].id = i;
		list_lm.features[i].list_lm_observed_simul = list_lm_.features[i].list_lm_observed_simul;
	}
}
bool PlanarLandmarkEKF::RemoveUnavailableLM::CheckNormalIsInward_(const Eigen::Vector3d& Ng)
{
	Eigen::Vector3d VerticalPosition = X_.segment(0, 3).dot(Ng)/Ng.dot(Ng)*Ng;
	double dot = VerticalPosition.dot(Ng);
	if(dot<0)	return true;
	else{
		double dist_wall = Ng.norm();
		double dist_robot = VerticalPosition.norm();
		if(dist_robot<dist_wall)	return true;
		else	return false;
	}
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "planar_landmark_ekf");
	std::cout << "Planar Landmark EKF" << std::endl;
	
	PlanarLandmarkEKF planar_landmark_ekf;
	ros::spin();
}
