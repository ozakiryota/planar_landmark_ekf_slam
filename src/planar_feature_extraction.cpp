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
		/*objects*/
		pcl::visualization::PCLVisualizer viewer {"Planar Feature Extraction"};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> clusters;
		pcl::PointCloud<pcl::PointNormal>::Ptr vis_features {new pcl::PointCloud<pcl::PointNormal>};
		planar_landmark_ekf_slam::PlanarFeatureArray features;
		/*parameters*/
		double cluster_tolerance;
		int min_cluster_size;
	public:
		PlanarFeatureExtraction();
		void CallbackNC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void ClearArray(void);
		void Clustering(void);
		static bool CustomCondition(const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float squaredDistance);
		Eigen::Vector4d ComputeAverageNormal(pcl::PointIndices indices);
		void Visualization(void);
		void Publication(void);
};

PlanarFeatureExtraction::PlanarFeatureExtraction()
	:nhPrivate("~")
{
	sub_nc = nh.subscribe("/normals", 1, &PlanarFeatureExtraction::CallbackNC, this);
	pub_features = nh.advertise<planar_landmark_ekf_slam::PlanarFeatureArray>("/features", 1);
	
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(1.0, "axis");
	viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

	nhPrivate.param("cluster_tolerance", cluster_tolerance, 0.1);
	nhPrivate.param("min_cluster_size", min_cluster_size, 100);
	std::cout << "cluster_tolerance = " << cluster_tolerance << std::endl;
	std::cout << "min_cluster_size = " << min_cluster_size << std::endl;
}

void PlanarFeatureExtraction::CallbackNC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	/* std::cout << "CALLBACK PC" << std::endl; */

	pcl::fromROSMsg(*msg, *normals);
	std::cout << "==========" << std::endl;
	std::cout << "normals->points.size() = " << normals->points.size() << std::endl;
	features.header = msg->header;

	ClearArray();
	Clustering();
	Visualization();
	Publication();
}

void PlanarFeatureExtraction::ClearArray(void)
{
	features.features.clear();
	clusters.clear();
	vis_features->points.clear();
}

void PlanarFeatureExtraction::Clustering(void)
{
	double time_start = ros::Time::now().toSec();

	/*clustering*/
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec(true);
	cec.setInputCloud(normals);
	cec.setConditionFunction(&CustomCondition);
	cec.setClusterTolerance(cluster_tolerance);
	cec.setMinClusterSize(min_cluster_size);
	cec.setMaxClusterSize(normals->points.size());
	cec.segment(cluster_indices);

	std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;

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
		Eigen::Vector4d AveNormal = ComputeAverageNormal(cluster_indices[i]);
		/* #<{(|compute centroid|)}># */
		/* Eigen::Vector4f centroid; */
		/* pcl::compute3DCentroid(*normals, cluster_indices[i], centroid); */
		/*input*/
		planar_landmark_ekf_slam::PlanarFeature tmp_feature;
		tmp_feature.point_local.x = -fabs(AveNormal(3))*AveNormal(0);
		tmp_feature.point_local.y = -fabs(AveNormal(3))*AveNormal(1);
		tmp_feature.point_local.z = -fabs(AveNormal(3))*AveNormal(2);
		tmp_feature.min_local.x = Min[0];
		tmp_feature.min_local.y = Min[1];
		tmp_feature.min_local.z = Min[2];
		tmp_feature.max_local.x = Max[0];
		tmp_feature.max_local.y = Max[1];
		tmp_feature.max_local.z = Max[2];
		tmp_feature.cluster_size = cluster_indices[i].indices.size();
		features.features.push_back(tmp_feature);
		/*input for visualization*/
		pcl::PointNormal tmp_normal;
		tmp_normal.x = 0;
		tmp_normal.y = 0;
		tmp_normal.z = 0;
		tmp_normal.data_n[0] = -fabs(AveNormal(3))*AveNormal(0);
		tmp_normal.data_n[1] = -fabs(AveNormal(3))*AveNormal(1);
		tmp_normal.data_n[2] = -fabs(AveNormal(3))*AveNormal(2);
		vis_features->points.push_back(tmp_normal);
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

	const double threshold_angle = 1.0;	//[deg]
	if(angle/M_PI*180.0 < threshold_angle)	return true;
	else	return false;
}

Eigen::Vector4d PlanarFeatureExtraction::ComputeAverageNormal(pcl::PointIndices indices)
{
	Eigen::Vector4d Ave(0.0, 0.0, 0.0, 0.0);
	for(size_t i=0;i<indices.indices.size();++i){
		Eigen::Vector4d N(
			normals->points[indices.indices[i]].data_n[0],
			normals->points[indices.indices[i]].data_n[1],
			normals->points[indices.indices[i]].data_n[2],
			normals->points[indices.indices[i]].data_n[3]
		);
		Ave += N;
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
	viewer.addPointCloudNormals<pcl::PointNormal>(vis_features, 1, 1.0, "vis_features");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "vis_features");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "vis_features");

	viewer.spinOnce();
}

void PlanarFeatureExtraction::Publication(void)
{
	pub_features.publish(features);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "planar_feature_extraction");
	
	PlanarFeatureExtraction planar_feature_extraction;

	ros::spin();
}
