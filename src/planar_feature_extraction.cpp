#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>

#include "planar_landmark_ekf_slam/PlanarFeature.h"
#include "planar_landmark_ekf_slam/PlanarFeatureArray.h"

class ConditionalEuclideanClustering{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_nc;
		/*pcl objects*/
		pcl::visualization::PCLVisualizer viewer {"Planar Feature Extraction"};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> clusters;
		/*parameters*/
		double cluster_tolerance;
		int min_cluster_size;
	public:
		ConditionalEuclideanClustering();
		void CallbackNC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void Clustering(void);
		static bool CustomCondition(const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float squaredDistance);
		void Visualization(void);
};

ConditionalEuclideanClustering::ConditionalEuclideanClustering()
	:nhPrivate("~")
{
	sub_nc = nh.subscribe("/normals", 1, &ConditionalEuclideanClustering::CallbackNC, this);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(1.0, "axis");
	viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

	nhPrivate.param("cluster_tolerance", cluster_tolerance, 0.1);
	nhPrivate.param("min_cluster_size", min_cluster_size, 100);
	std::cout << "cluster_tolerance = " << cluster_tolerance << std::endl;
	std::cout << "min_cluster_size = " << min_cluster_size << std::endl;
}

void ConditionalEuclideanClustering::CallbackNC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	/* std::cout << "CALLBACK PC" << std::endl; */

	pcl::fromROSMsg(*msg, *normals);
	std::cout << "==========" << std::endl;
	std::cout << "normals->points.size() = " << normals->points.size() << std::endl;

	clusters.clear();
	Clustering();
	Visualization();
}

void ConditionalEuclideanClustering::Clustering(void)
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
		std::cout << i << ": Min  = " << std::endl << Min << std::endl;
		std::cout << i << ": Max  = " << std::endl << Max << std::endl;
		/*compute centroid*/
		Eigen::Vector4f centroid;
		pcl::compute3DCentroid(*normals, cluster_indices[i], centroid);
		/*input*/

		/*extract for visualization*/
		pcl::PointCloud<pcl::PointNormal>::Ptr tmp_cluster (new pcl::PointCloud<pcl::PointNormal>);
		pcl::PointIndices::Ptr tmp_clustered_indices (new pcl::PointIndices);	//Can this be skipped?
		*tmp_clustered_indices = cluster_indices[i];
		ei.setIndices(tmp_clustered_indices);
		ei.filter(*tmp_cluster);
		clusters.push_back(tmp_cluster);
	}

	std::cout << "clustering time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
}

bool ConditionalEuclideanClustering::CustomCondition(const pcl::PointNormal& seedPoint, const pcl::PointNormal& candidatePoint, float squaredDistance)
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

void ConditionalEuclideanClustering::Visualization(void)
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

	viewer.spinOnce();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "euclidean_clustering");
	
	ConditionalEuclideanClustering euclidean_clustering;

	ros::spin();
}
