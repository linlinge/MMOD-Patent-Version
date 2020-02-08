#include "DataMining.h"
double DataMining::erf(double x)
{
	double a1=0.278393;
	double a2=0.230389;
	double a3=0.000972;
	double a4=0.078108;
	double m=1+a1*x+a2*pow(x,2)+a3*pow(x,3)+a4*pow(x,4);
	return (1-1.0/pow(m,4));
}

DataMining::DataMining(pcl::PointCloud<PointType>::Ptr cloud)
{
	// set cloud
	cloud_=cloud;
	// establish kdtree_
	kdtree_=pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>());
	kdtree_->setInputCloud(cloud_);
}

void DataMining::LOOP(int k,double threshould)
{
	// Resize Scores
	scores_.resize(cloud_->points.size());
	sigma_.resize(cloud_->points.size());
	plof_.resize(cloud_->points.size());
	rst_.Resize(cloud_->points.size());

	// Step 01: Calculate sigma
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		// find k-nearest neighours
		vector<int> pointIdxNKNSearch(k+1);
		vector<float> pointNKNSquaredDistance(k+1);
		kdtree_->nearestKSearch (cloud_->points[i], k+1, pointIdxNKNSearch, pointNKNSquaredDistance);
		// cout<<cloud->points[i]<<endl;
		double sum=0;
		for(int j=1;j<k+1;j++){
			sum+=pointNKNSquaredDistance[j];
		}
		sum=sum/k;
		sigma_[i]=sqrt(sum);
	}
	
	// Step 02: calculate mean
	double mean=0;
	// #pragma omp parallel for
	for (int i = 0; i < cloud_->points.size(); i++){        
        vector<int> pointIdxNKNSearch(k+1);
		vector<float> pointNKNSquaredDistance(k+1);
		kdtree_->nearestKSearch (cloud_->points[i], k+1, pointIdxNKNSearch, pointNKNSquaredDistance);
        double sum = 0;
        for (int j = 1; j < k+1; j++)
          sum += sigma_[pointIdxNKNSearch[j]];
        sum /= k;
        plof_[i] = sigma_[i] / sum  - 1.0f;
		rst_.records_[i].item1_=plof_[i];
		
        mean += plof_[i] * plof_[i];
    }
	mean=mean/cloud_->points.size();
	mean=sqrt(mean);

	// Step 03: caculate score
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		double value = plof_[i] / (mean * sqrt(2.0f));
		// rst_.records_[i].item1_=value;

        double dem = 1.0 + 0.278393 * value;
        dem += 0.230389 * value * value;
        dem += 0.000972 * value * value * value;
        dem += 0.078108 * value * value * value * value;
        double op = std::max(0.0, 1.0 - 1.0 / dem);
        scores_[i] = op;
	}

	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		// cout<<i<<endl;
		if(scores_[i]>threshould){
			cloud_->points[i].r=255;
			cloud_->points[i].g=0;
			cloud_->points[i].b=0;
		}
	}
	pcl::io::savePLYFileASCII("Result/result.ply",*cloud_);
}

double GaussianKernel(double u, int n)
{
	return 1/sqrt(0.2*M_PI)*exp(-pow(u,2)/0.2);
}

// Calculate entropy with kernel density estimation (KDE)
void EntropyWithKDE(pcl::PointCloud<PointType>::Ptr cloud)
{
	// Step 1: Init Parameters
	int K=32;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);
	vector<double> entropy;

	// Step 2: entropy
	for(int i=0;i<cloud->points.size();i++){
		// Step 2-1: Define Parameters 
		vector<int> idx(K);
		vector<float> dist(K);
		// pcl::PointCloud<PointType>::Ptr ctmp(new pcl::PointCloud<PointType>);

		// Step 2-2: Calculate entropy
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		double dtmp=0;
		double h=0.5;
		for(int j=0;j<K;j++){
			dtmp+=GaussianKernel(dist[j]/h,K);
		}
		dtmp=dtmp/K/h;

		// Step 2-3: Rendering
		// cout<<dtmp<<endl;
		cloud->points[i].r=dtmp*255;
		cloud->points[i].g=0;
		cloud->points[i].b=0;
	}
}

