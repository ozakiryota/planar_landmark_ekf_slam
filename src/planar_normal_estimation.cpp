#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <omp.h>

class PlanarNormalEstimation{
	private:
		/*node handle*/
		ros::NodeHandle nh;
		ros::NodeHandle nhPrivate;
		/*subscribe*/
		ros::Subscriber sub_pc;
		/*publish*/
		ros::Publisher pub_nc;
		ros::Publisher pub_selected_nc;
		ros::Publisher pub_vis_gauss;
		/*pcl*/
		pcl::visualization::PCLVisualizer viewer {"Planar Normal Estimation"};
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointNormal>::Ptr normals {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointNormal>::Ptr selected_normals {new pcl::PointCloud<pcl::PointNormal>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr d_gaussian_sphere {new pcl::PointCloud<pcl::PointXYZ>};
		pcl::PointCloud<pcl::PointXYZ>::Ptr selected_d_gaussian_sphere {new pcl::PointCloud<pcl::PointXYZ>};
		/*objects*/
		Eigen::Vector3f Gvector{0.0, 0.0, -1.0};	//tmp
		/*parameters*/
		int skip;
		double search_radius_ratio;
		double min_search_radius;
		/* double max_search_radius; */
		int threshold_num_neighborpoints;
		bool mode_remove_ground;
		bool mode_open_viewer;
		bool mode_selection;
	public:
		PlanarNormalEstimation();
		void CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg);
		void ClearPC(void);
		void Computation(void);
		double Getdepth(pcl::PointXYZ point);
		std::vector<int> KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius);
		bool JudgeForSelecting(const Eigen::Vector4f& plane_parameters, std::vector<int> indices);
		double AngleBetweenVectors(const Eigen::Vector3f& V1, const Eigen::Vector3f& V2);
		double ComputeFittingError(const Eigen::Vector4f& N, std::vector<int> indices);
		void Visualization(void);
		void Publication(void);
};

PlanarNormalEstimation::PlanarNormalEstimation()
	:nhPrivate("~")
{
	sub_pc = nh.subscribe("/velodyne_points", 1, &PlanarNormalEstimation::CallbackPC, this);
	pub_nc = nh.advertise<sensor_msgs::PointCloud2>("/normals", 1);
	pub_selected_nc = nh.advertise<sensor_msgs::PointCloud2>("/selected_normals", 1);
	pub_vis_gauss = nh.advertise<sensor_msgs::PointCloud2>("/selected_d_gaussian_sphere", 1);
	viewer.setBackgroundColor(1, 1, 1);
	viewer.addCoordinateSystem(1.0, "axis");
	// viewer.setCameraPosition(-30.0, 0.0, 10.0, 0.0, 0.0, 1.0);
	viewer.setCameraPosition(0.0, 0.0, 35.0, 0.0, 0.0, 0.0);

	nhPrivate.param("skip", skip, 3);
	nhPrivate.param("search_radius_ratio", search_radius_ratio, 0.09);
	nhPrivate.param("min_search_radius", min_search_radius, 0.1);
	/* nhPrivate.param("max_search_radius", max_search_radius, 2.0); */
	nhPrivate.param("threshold_num_neighborpoints", threshold_num_neighborpoints, 10);
	nhPrivate.param("mode_remove_ground", mode_remove_ground, false);
	nhPrivate.param("mode_open_viewer", mode_open_viewer, true);
	nhPrivate.param("mode_selection", mode_selection, true);
	std::cout << "skip = " << skip << std::endl;
	std::cout << "search_radius_ratio = " << search_radius_ratio << std::endl;
	std::cout << "min_search_radius = " << min_search_radius << std::endl;
	/* std::cout << "max_search_radius = " << max_search_radius << std::endl; */
	std::cout << "mode_remove_ground = " << (bool)mode_remove_ground << std::endl;
	std::cout << "mode_open_viewer = " << (bool)mode_open_viewer << std::endl;
	std::cout << "mode_selection = " << (bool)mode_selection << std::endl;

	if(!mode_open_viewer)	viewer.close();
}

void PlanarNormalEstimation::CallbackPC(const sensor_msgs::PointCloud2ConstPtr &msg)
{
	/* std::cout << "CALLBACK PC" << std::endl; */

	pcl::fromROSMsg(*msg, *cloud);
	std::cout << "==========" << std::endl;
	std::cout << "cloud->points.size() = " << cloud->points.size() << std::endl;
	ClearPC();

	kdtree.setInputCloud(cloud);
	Computation();

	if(mode_open_viewer)	Visualization();
	Publication();
}

void PlanarNormalEstimation::ClearPC(void)
{
	normals->points.clear();
	selected_normals->points.clear();
	d_gaussian_sphere->points.clear();
	selected_d_gaussian_sphere->points.clear();
}

void PlanarNormalEstimation::Computation(void)
{
	std::cout << "omp_get_max_threads() = " << omp_get_max_threads() << std::endl;

	double time_start = ros::Time::now().toSec();

	normals->points.resize((cloud->points.size()-1)/skip + 1);
	d_gaussian_sphere->points.resize((cloud->points.size()-1)/skip + 1);
	std::vector<bool> extract_indices((cloud->points.size()-1)/skip + 1, false);

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for(size_t i=0;i<cloud->points.size();i+=skip){
		size_t normal_index = i/skip;
		/*search neighbor points*/
		double laser_distance = Getdepth(cloud->points[i]);
		double search_radius = search_radius_ratio*laser_distance;
		if(search_radius<min_search_radius)	search_radius = min_search_radius;
		// if(search_radius>max_search_radius)	search_radius = max_search_radius;
		std::vector<int> indices = KdtreeSearch(cloud->points[i], search_radius);
		/*compute normal*/
		float curvature;
		Eigen::Vector4f plane_parameters;
		pcl::computePointNormal(*cloud, indices, plane_parameters, curvature);
		/*judge*/
		if(mode_selection)	extract_indices[normal_index] = JudgeForSelecting(plane_parameters, indices);
		else	extract_indices[normal_index] = true;
		/*input*/
		normals->points[normal_index].x = cloud->points[i].x;
		normals->points[normal_index].y = cloud->points[i].y;
		normals->points[normal_index].z = cloud->points[i].z;
		normals->points[normal_index].data_n[0] = plane_parameters(0);
		normals->points[normal_index].data_n[1] = plane_parameters(1);
		normals->points[normal_index].data_n[2] = plane_parameters(2);
		normals->points[normal_index].data_n[3] = plane_parameters(3);
		normals->points[normal_index].curvature = curvature;
		flipNormalTowardsViewpoint(cloud->points[i], 0.0, 0.0, 0.0, normals->points[normal_index].normal_x, normals->points[normal_index].normal_y, normals->points[normal_index].normal_z);
		/*Gauss map*/
		d_gaussian_sphere->points[normal_index].x = -fabs(normals->points[normal_index].data_n[3])*normals->points[normal_index].data_n[0];
		d_gaussian_sphere->points[normal_index].y = -fabs(normals->points[normal_index].data_n[3])*normals->points[normal_index].data_n[1];
		d_gaussian_sphere->points[normal_index].z = -fabs(normals->points[normal_index].data_n[3])*normals->points[normal_index].data_n[2];
	}
	if(mode_selection){
		for(size_t i=0;i<extract_indices.size();i++){
			if(extract_indices[i]){
				/*selected normals*/
				selected_normals->points.push_back(normals->points[i]);
				/*selected d-gaussian shpere*/
				selected_d_gaussian_sphere->points.push_back(d_gaussian_sphere->points[i]);
			}
		}
	}
	else{
		selected_normals = normals;
		selected_d_gaussian_sphere = d_gaussian_sphere;
	}

	std::cout << "computation time [s] = " << ros::Time::now().toSec() - time_start << std::endl;
	std::cout << "selected_normals->points.size() = " << selected_normals->points.size() << "(" << selected_normals->points.size()/(double)cloud->points.size()*100.0 << " %)" << std::endl;
}

double PlanarNormalEstimation::Getdepth(pcl::PointXYZ point)
{
	double depth = sqrt(
		point.x*point.x
		+ point.y*point.y
		+ point.z*point.z
	);
	return depth;
}

std::vector<int> PlanarNormalEstimation::KdtreeSearch(pcl::PointXYZ searchpoint, double search_radius)
{
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	if(kdtree.radiusSearch(searchpoint, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance)<=0)	std::cout << "kdtree error" << std::endl;
	return pointIdxRadiusSearch; 
}

bool PlanarNormalEstimation::JudgeForSelecting(const Eigen::Vector4f& plane_parameters, std::vector<int> indices)
{
	/*threshold*/
	const double threshold_angle = 30.0;	//[deg]
	const double threshold_fitting_error = 0.01;	//[m]

	/*number of neighbor-points*/
	if(indices.size() < threshold_num_neighborpoints)	return false;
	/*nan*/
	if(std::isnan(plane_parameters(0)) || std::isnan(plane_parameters(1)) || std::isnan(plane_parameters(2)))	return false;
	/*angle*/
	if(mode_remove_ground){
		if(fabs(AngleBetweenVectors(plane_parameters.segment(0, 3), Gvector)-M_PI/2.0)>threshold_angle/180.0*M_PI)	return false;
	}
	/*fitting error*/
	if(ComputeFittingError(plane_parameters, indices) > threshold_fitting_error)	return false;
	/*pass*/
	return true;
}

double PlanarNormalEstimation::AngleBetweenVectors(const Eigen::Vector3f& V1, const Eigen::Vector3f& V2)
{
	double angle = acos(V1.dot(V2)/V1.norm()/V2.norm());
	return angle;
}

double PlanarNormalEstimation::ComputeFittingError(const Eigen::Vector4f& N, std::vector<int> indices)
{
	double ave_fitting_error = 0.0;
	for(size_t i=0;i<indices.size();++i){
		Eigen::Vector3f P(
			cloud->points[indices[i]].x,
			cloud->points[indices[i]].y,
			cloud->points[indices[i]].z
		);
		ave_fitting_error += fabs(N.segment(0, 3).dot(P) + N(3))/N.segment(0, 3).norm();
	}
	ave_fitting_error /= (double)indices.size();
	return ave_fitting_error;
}

void PlanarNormalEstimation::Visualization(void)
{
	viewer.removeAllPointClouds();

	/*cloud*/
	viewer.addPointCloud(cloud, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "cloud");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
	/*normals*/
	viewer.addPointCloudNormals<pcl::PointNormal>(normals, 1, 0.5, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "normals");
	/*selected normals*/
	viewer.addPointCloudNormals<pcl::PointNormal>(selected_normals, 1, 0.5, "selected_normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "selected_normals");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, "selected_normals");
	/*d-gaussian sphere*/
	viewer.addPointCloud(d_gaussian_sphere, "d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 0.0, "d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "d_gaussian_sphere");
	/*selected d-gaussian sphere*/
	viewer.addPointCloud(selected_d_gaussian_sphere, "selected_d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "selected_d_gaussian_sphere");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "selected_d_gaussian_sphere");
	
	viewer.spinOnce();
}

void PlanarNormalEstimation::Publication(void)
{
	/*nc*/
	normals->header.stamp = cloud->header.stamp;
	normals->header.frame_id = cloud->header.frame_id;
	sensor_msgs::PointCloud2 nc_pub;
	pcl::toROSMsg(*normals, nc_pub);
	pub_nc.publish(nc_pub);	
	/*nc*/
	if(mode_selection){
		selected_normals->header.stamp = cloud->header.stamp;
		selected_normals->header.frame_id = cloud->header.frame_id;
		sensor_msgs::PointCloud2 selected_nc_pub;
		pcl::toROSMsg(*selected_normals, selected_nc_pub);
		pub_selected_nc.publish(selected_nc_pub);
	}
	/*gauss*/
	selected_d_gaussian_sphere->header.stamp = cloud->header.stamp;
	selected_d_gaussian_sphere->header.frame_id = cloud->header.frame_id;
	sensor_msgs::PointCloud2 gauss_pub;
	pcl::toROSMsg(*selected_d_gaussian_sphere, gauss_pub);
	pub_vis_gauss.publish(gauss_pub);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "planar_normal_estimation");
	std::cout << "Planar Normal Estimation" << std::endl;
	
	PlanarNormalEstimation planar_normal_estimation;

	ros::spin();
}
