#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/common/common.h>
/* #include <pcl/common/centroid.h> */
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>

#include "planar_landmark_ekf_slam/PlanarFeature.h"
#include "planar_landmark_ekf_slam/PlanarFeatureArray.h"

class PlanarFeatureExtraction{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_nc;
		/*publish*/
		ros::Publisher pub_features;
		ros::Publisher pub_pc;
		/*objects*/
		pcl::visualization::PCLVisualizer viewer {"Planar Feature Extraction"};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> clusters;
		pcl::PointCloud<pcl::PointXYZ>::Ptr features_p {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointNormal>::Ptr features_n {new pcl::PointCloud<pcl::PointNormal>};
		planar_landmark_ekf_slam::PlanarFeatureArray feature_array;
		/*parameters*/
		double ratio_depth_tolerance;
		double min_tolerance;
		double max_tolerance;
		int min_cluster_size;
		int max_cluster_size;
		double threshold_angle;	//[deg]
	public:
		PlanarFeatureExtraction();
		void CallbackNC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void ClearArray(void);
		void Clustering(void);
		double ComputeTolerance(const pcl::PointNormal& point);
		bool CustomCondition(const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float squaredDistance);
		// Eigen::Vector4d ComputeAverageNormal_(pcl::PointIndices indices);
		Eigen::Vector3d ComputeAverageNormal(const pcl::PointIndices& indices);
		void Visualization(void);
		void Publication(void);
};

PlanarFeatureExtraction::PlanarFeatureExtraction()
	:nhPrivate("~")
{
	sub_nc = nh.subscribe("/normals", 1, &PlanarFeatureExtraction::CallbackNC, this);
	pub_features = nh.advertise<planar_landmark_ekf_slam::PlanarFeatureArray>("/features", 1);
	pub_pc = nh.advertise<sensor_msgs::PointCloud2>("/observation", 1);
	
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(1.0, "axis");
	/* viewer.setCameraPosition(-30.0, 0.0, 20.0, 0.0, 0.0, 1.0); */
	viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

	nhPrivate.param("ratio_depth_tolerance", ratio_depth_tolerance, 0.05);
	std::cout << "ratio_depth_tolerance = " << ratio_depth_tolerance << std::endl;
	nhPrivate.param("min_tolerance", min_tolerance, 0.1);
	std::cout << "min_tolerance = " << min_tolerance << std::endl;
	nhPrivate.param("max_tolerance", max_tolerance, 0.5);
	std::cout << "max_tolerance = " << max_tolerance << std::endl;
	nhPrivate.param("min_cluster_size", min_cluster_size, 100);
	std::cout << "min_cluster_size = " << min_cluster_size << std::endl;
	nhPrivate.param("threshold_angle", threshold_angle, 1.0);
	std::cout << "threshold_angle = " << threshold_angle << std::endl;
}

void PlanarFeatureExtraction::CallbackNC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	/* std::cout << "CALLBACK PC" << std::endl; */

	pcl::fromROSMsg(*msg, *normals);
	std::cout << "==========" << std::endl;
	std::cout << "normals->points.size() = " << normals->points.size() << std::endl;
	feature_array.header = msg->header;

	ClearArray();
	Clustering();
	Visualization();
	if(!feature_array.features.empty())	Publication();
}

void PlanarFeatureExtraction::ClearArray(void)
{
	clusters.clear();
	features_p->points.clear();
	features_n->points.clear();
	feature_array.features.clear();
}

void PlanarFeatureExtraction::Clustering(void)
{
	double time_start = ros::Time::now().toSec();

	/*searching config*/
	pcl::KdTreeFLANN<pcl::PointNormal> kdtree;
	kdtree.setInputCloud(normals);
	max_cluster_size = normals->points.size();
	/*objects*/
	std::vector<pcl::PointIndices> cluster_indices;
	std::vector<bool> processed(normals->points.size(), false);
	std::vector<int> nn_indices;
	std::vector<float> nn_distances;
	/*clustering*/
	for(size_t i=0;i<normals->points.size();++i){
		if(processed[i])	continue;
		/*set seed*/
		std::vector<int> seed_queue;
		int sq_idx = 0;
		seed_queue.push_back(i);
		processed[i] = true;
		/*clustering*/
		while(sq_idx < seed_queue.size()){
			/*search*/
			double tolerance = ComputeTolerance(normals->points[seed_queue[sq_idx]]);
			int ret = kdtree.radiusSearch(normals->points[seed_queue[sq_idx]], tolerance, nn_indices, nn_distances);
			if(ret == -1){
				PCL_ERROR("[pcl::extractEuclideanClusters] Received error code -1 from radiusSearch\n");
				exit(0);
			}
			/*check*/
			for(size_t j=0;j<nn_indices.size();++j){
				if(nn_indices[j]==-1 || processed[nn_indices[j]])	continue;
				if(CustomCondition(normals->points[seed_queue[sq_idx]], normals->points[nn_indices[j]], nn_distances[j])){
					seed_queue.push_back(nn_indices[j]);
					processed[nn_indices[j]] = true;
				}
			}
			sq_idx++;
		}
		/*judge*/
		if(seed_queue.size()>=min_cluster_size && seed_queue.size()<=max_cluster_size){
			pcl::PointIndices tmp_indices;
			tmp_indices.indices = seed_queue;
			cluster_indices.push_back(tmp_indices);
		}
	}
	std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;
	/*extraction*/
	pcl::ExtractIndices<pcl::PointNormal> ei;
	ei.setInputCloud(normals);
	ei.setNegative(false);
	for(size_t i=0;i<cluster_indices.size();i++){
		/*get min-max*/
		Eigen::Vector4f Min;
		Eigen::Vector4f Max;		
		getMinMax3D(*normals, cluster_indices[i], Min, Max);
		/* std::cout << i << ": Min = " << std::endl << Min << std::endl; */
		/* std::cout << i << ": Max = " << std::endl << Max << std::endl; */
		/*average normal*/
		// Eigen::Vector4d AveNormal_ = ComputeAverageNormal_(cluster_indices[i]);
		Eigen::Vector3d AveNormal = ComputeAverageNormal(cluster_indices[i]);
		/* #<{(|compute centroid|)}># */
		/* Eigen::Vector4f centroid; */
		/* pcl::compute3DCentroid(*normals, cluster_indices[i], centroid); */
		/*input*/
		planar_landmark_ekf_slam::PlanarFeature tmp_feature;
		/* tmp_feature.point_local.x = -AveNormal_(3)*AveNormal_(0); */
		/* tmp_feature.point_local.y = -AveNormal_(3)*AveNormal_(1); */
		/* tmp_feature.point_local.z = -AveNormal_(3)*AveNormal_(2); */
		tmp_feature.point_local.x = -AveNormal(0);
		tmp_feature.point_local.y = -AveNormal(1);
		tmp_feature.point_local.z = -AveNormal(2);
		tmp_feature.min_local.x = Min[0];
		tmp_feature.min_local.y = Min[1];
		tmp_feature.min_local.z = Min[2];
		tmp_feature.max_local.x = Max[0];
		tmp_feature.max_local.y = Max[1];
		tmp_feature.max_local.z = Max[2];
		tmp_feature.cluster_size = cluster_indices[i].indices.size();
		feature_array.features.push_back(tmp_feature);
		/*input for visualization*/
		pcl::PointNormal tmp_normal;
		tmp_normal.x = 0;
		tmp_normal.y = 0;
		tmp_normal.z = 0;
		/* tmp_normal.data_n[0] = -AveNormal_(3)*AveNormal_(0); */
		/* tmp_normal.data_n[1] = -AveNormal_(3)*AveNormal_(1); */
		/* tmp_normal.data_n[2] = -AveNormal_(3)*AveNormal_(2); */
		tmp_normal.data_n[0] = -AveNormal(0);
		tmp_normal.data_n[1] = -AveNormal(1);
		tmp_normal.data_n[2] = -AveNormal(2);
		features_n->points.push_back(tmp_normal);
		/*input for visualization*/
		pcl::PointXYZ tmp_point;
		/* tmp_point.x = -AveNormal_(3)*AveNormal_(0); */
		/* tmp_point.y = -AveNormal_(3)*AveNormal_(1); */
		/* tmp_point.z = -AveNormal_(3)*AveNormal_(2); */
		tmp_point.x = -AveNormal(0);
		tmp_point.y = -AveNormal(1);
		tmp_point.z = -AveNormal(2);
		features_p->points.push_back(tmp_point);
		/*extraction for visualization*/
		pcl::PointCloud<pcl::PointNormal>::Ptr tmp_cluster (new pcl::PointCloud<pcl::PointNormal>);
		pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);
		*tmp_clustered_indices = cluster_indices[i];
		ei.setIndices(tmp_clustered_indices);
		ei.filter(*tmp_cluster);
		clusters.push_back(tmp_cluster);
	}

	std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
}

double PlanarFeatureExtraction::ComputeTolerance(const pcl::PointNormal& point)
{
	double depth = sqrt(
		point.x * point.x
		+ point.y * point.y
		+ point.z * point.z
	);
	double tolerance = ratio_depth_tolerance*depth;
	if(tolerance < min_tolerance)	tolerance = min_tolerance;
	if(tolerance > max_tolerance)	tolerance = max_tolerance;

	return tolerance;
}

bool PlanarFeatureExtraction::CustomCondition(const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float squaredDistance)
{
	Eigen::Vector3d N1(
		seedPoint.normal_x,
		seedPoint.normal_y,
		seedPoint.normal_z
	);
	Eigen::Vector3d N2(
		candidatePoint.normal_x,
		candidatePoint.normal_y,
		candidatePoint.normal_z
	);
	double angle = acos(N1.dot(N2)/N1.norm()/N2.norm());

	if(angle/M_PI*180.0 < threshold_angle)	return true;
	else	return false;
}

/* Eigen::Vector4d PlanarFeatureExtraction::ComputeAverageNormal_(pcl::PointIndices indices) */
/* { */
/* 	Eigen::Vector4d Ave(0.0, 0.0, 0.0, 0.0); */
/* 	for(size_t i=0;i<indices.indices.size();++i){ */
/* 		Eigen::Vector4d N( */
/* 			normals->points[indices.indices[i]].normal_x, */
/* 			normals->points[indices.indices[i]].normal_y, */
/* 			normals->points[indices.indices[i]].normal_z, */
/* 			fabs(normals->points[indices.indices[i]].data_n[3]) */
/* 		); */
/* 		Ave += N; */
/* 	} */
/* 	Ave /= (double)indices.indices.size(); */
/*  */
/* 	return Ave; */
/* } */
Eigen::Vector3d PlanarFeatureExtraction::ComputeAverageNormal(const pcl::PointIndices& indices)
{
	Eigen::Vector3d Ave(0.0, 0.0, 0.0);
	for(size_t i=0;i<indices.indices.size();++i){
		Eigen::Vector3d N(
			fabs(normals->points[indices.indices[i]].data_n[3]) * normals->points[indices.indices[i]].normal_x,
			fabs(normals->points[indices.indices[i]].data_n[3]) * normals->points[indices.indices[i]].normal_y,
			fabs(normals->points[indices.indices[i]].data_n[3]) * normals->points[indices.indices[i]].normal_z
		);
		Ave += N;
		/* std::cout << "N" << i << ": (" << N(0) << ", " << N(1) << ", " << N(2) << std::endl; */
	}
	Ave /= (double)indices.indices.size();

	return Ave;
}

void PlanarFeatureExtraction::Visualization(void)
{
	viewer.removeAllPointClouds();

	/*normals*/
	viewer.addPointCloudNormals<pcl::PointNormal>(normals, 1, 0.5, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 0.5, "normals");
	/*clusters*/
	double rgb[3] = {};
	const int channel = 3;
	const double step = ceil(pow(clusters.size()+2, 1.0/(double)channel));	//exept (000),(111)
	const double max = 1.0;
	for(size_t i=0;i<clusters.size();i++){
		std::string name = "cluster_" + std::to_string(i);
		rgb[0] += 1/step;
		for(int j=0;j<channel-1;j++){
			if(rgb[j]>max){
				rgb[j] -= max + 1/step;
				rgb[j+1] += 1/step;
			}
		}
		/* std::cout << "step = " << step << std::endl; */
		/* std::cout << name << ": (r,g,b) = " << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << std::endl; */
		/*input*/
		viewer.addPointCloudNormals<pcl::PointNormal>(clusters[i], 1, 0.5, name);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, rgb[0], rgb[1], rgb[2], name);
		viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, name);
	}
	/*features*/
	viewer.addPointCloudNormals<pcl::PointNormal>(features_n, 1, 1.0, "features_n");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "features_n");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "features_n");

	viewer.spinOnce();
}

void PlanarFeatureExtraction::Publication(void)
{
	/*features*/
	pub_features.publish(feature_array);
	/*d-gaussian sphere (visualization)*/
	features_p->header.stamp = normals->header.stamp;
	features_p->header.frame_id = normals->header.frame_id;
	sensor_msgs::PointCloud2 observation;
	pcl::toROSMsg(*features_p, observation);
	pub_pc.publish(observation);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "planar_feature_extraction");
	
	PlanarFeatureExtraction planar_feature_extraction;

	ros::spin();
}
