#include "ImprovedLoop.h"
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>

void ImprovedLoop::StatisticCentreAndCentroid(pcl::PointCloud<PointType>::Ptr cloud)
{    
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    double mean_dist=ComputeMeanDistance(cloud);
    vector<double> stcc;
    stcc.resize(cloud->points.size());
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx;
        vector<float> dist;
        kdtree->radiusSearch(cloud->points[i], 8*mean_dist, idx, dist);

        // Calculate centroid
        double cx,cy,cz;
        cx=cy=cz=0;
        for(int j=0;j<idx.size();j++){
            cx+=cloud->points[idx[j]].x;
            cy+=cloud->points[idx[j]].y;
            cz+=cloud->points[idx[j]].z;
        }
        cx=cx/idx.size();
        cy=cy/idx.size();
        cz=cz/idx.size();

        // calculate distance
        stcc[i]=sqrt(pow(cx-cloud->points[i].x,2)+pow(cy-cloud->points[i].y,2)+pow(cz-cloud->points[i].z,2));
    }

    ofstream fout("stcc.csv");
    for(int i=0;i<cloud->points.size();i++){
        fout<<stcc[i]<<endl;
    }
    fout.close();
}

double ImprovedLoop::erf(double x)
{
	double a1=0.278393;
	double a2=0.230389;
	double a3=0.000972;
	double a4=0.078108;
	double m=1+a1*x+a2*pow(x,2)+a3*pow(x,3)+a4*pow(x,4);
	return (1-1.0/pow(m,4));
}

void ImprovedLoop::Init(pcl::PointCloud<PointType>::Ptr cloud)
{
	// set cloud
	cloud_=cloud;
	// establish kdtree_
	kdtree_=pcl::search::KdTree<PointType>::Ptr(new pcl::search::KdTree<PointType>());
	kdtree_->setInputCloud(cloud_);
}

void ImprovedLoop::ILoop(int k,double threshould)
{
	// Resize Scores
	scores_.resize(cloud_->points.size());
	sigma_.resize(cloud_->points.size());
	plof_.resize(cloud_->points.size());

	// Step 01: Calculate sigma
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		// step 1-1: find k-nearest neighours
		vector<int> idx(k+1);
		vector<float> dist(k+1);
		kdtree_->nearestKSearch(cloud_->points[i], k+1, idx, dist);
        pcl::PointCloud<PointType>::Ptr ptmp(new pcl::PointCloud<PointType>);
        for(int j=0;j<idx.size();j++){
            ptmp->points.push_back(cloud_->points[idx[j]]);
        }

        // step 1-2: get projection vector
        EvalAndEvec vv(ptmp);
        // vv.MajorMinor(ptmp);
        Eigen::Vector3f v1;
        // v1=vv.minor_vec_;
		v1<<vv.eigenvector_[0].x,vv.eigenvector_[0].y,vv.eigenvector_[0].z;
		cloud_->points[i].normal_x=v1(0);
		cloud_->points[i].normal_y=v1(1);
		cloud_->points[i].normal_z=v1(2);

        // step 1-3: calculate sum	
		double sum=0;
        PointType p1=cloud_->points[i];
		for(int j=1;j<k+1;j++){
            PointType p2=cloud_->points[idx[j]];
            Eigen::Vector3f v2;
            v2<<p2.x,p2.y,p2.z;
            auto prj=v1.transpose()*v2/v1.norm();
			// cout<<prj<<endl;
			sum+=abs(prj(0));
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
        mean += plof_[i] * plof_[i];
    }
	mean=mean/cloud_->points.size();
	mean=sqrt(mean);

	// Step 03: caculate score
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		double value = plof_[i] / (mean * sqrt(2.0f));
        double dem = 1.0 + 0.278393 * value;
        dem += 0.230389 * value * value;
        dem += 0.000972 * value * value * value;
        dem += 0.078108 * value * value * value * value;
        double op = std::max(0.0, 1.0 - 1.0 / dem);
        scores_[i] = op;
	}

	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		if(scores_[i]>threshould){
			cloud_->points[i].r=255;
			cloud_->points[i].g=0;
			cloud_->points[i].b=0;
		}
	}
	pcl::io::savePLYFileBinary("ImprovedLoop.ply",*cloud_);
}


void ImprovedLoop::Loop(int k,double threshould)
{
	// Resize Scores
	scores_.resize(cloud_->points.size());
	sigma_.resize(cloud_->points.size());
	plof_.resize(cloud_->points.size());

	// Step 01: Calculate sigma
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		// step 1-1: find k-nearest neighours
		vector<int> idx(k+1);
		vector<float> dist(k+1);
		kdtree_->nearestKSearch(cloud_->points[i], k+1, idx, dist);
        // step 1-3: calculate sum	
		double sum=0;
		for(int j=1;j<k+1;j++){
			sum+=dist[j];
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
        mean += plof_[i] * plof_[i];
    }
	mean=mean/cloud_->points.size();
	mean=sqrt(mean);

	// Step 03: caculate score
	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		double value = plof_[i] / (mean * sqrt(2.0f));
        double dem = 1.0 + 0.278393 * value;
        dem += 0.230389 * value * value;
        dem += 0.000972 * value * value * value;
        dem += 0.078108 * value * value * value * value;
        double op = std::max(0.0, 1.0 - 1.0 / dem);
        scores_[i] = op;
	}

	// #pragma omp parallel for
	for(int i=0;i<cloud_->points.size();i++){
		if(scores_[i]>threshould){
			cloud_->points[i].r=255;
			cloud_->points[i].g=0;
			cloud_->points[i].b=0;
		}
	}
	pcl::io::savePLYFileBinary("Loop.ply",*cloud_);
}


// Minor-Major Ratio, Central Limit Theorem
void CLT_MMR(pcl::PointCloud<PointType>::Ptr cloud)
{
	// Step 1: Init Parameters
	int K=32;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);

	// Step 2: calculate ratio=Minor/Major
	vector<double> mmr(cloud->points.size());
	vector<double> mmr_mean(cloud->points.size());
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// Step 2-1: Define Parameters 
		vector<int> idx(K);
		vector<float> dist(K);
		pcl::PointCloud<PointType>::Ptr ctmp(new pcl::PointCloud<PointType>);
		// Step 2-2: Establish cloud temp
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		for(int j=0;j<idx.size();j++){
			ctmp->points.push_back(cloud->points[idx[j]]);
		}
		// Step 2-3: get eigenvalue
		Eigen::Vector4f centroid;
		Eigen::Matrix3f covariance;
		pcl::compute3DCentroid(*ctmp, centroid);
		pcl::computeCovarianceMatrixNormalized(*ctmp, centroid, covariance);	
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);		
		Eigen::Vector3f eig_val = eigen_solver.eigenvalues();
		// Step 2-4: get ratio
		mmr[i]=eig_val(0)/eig_val(2);
	}

	// Step 3: calculate sample mean
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// step 3-1: find k-nearest neighbours
		vector<int> idx(K);
		vector<float> dist(K);
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		// step 3-2: calculate mean of mmr
		double cmean=0;
		for(int j=0;j<K;j++){
			cmean+=mmr[idx[j]];
		}
		cmean=cmean/K;
		// step 3-3: store cmean
		mmr_mean[i]=cmean;
	}

	// Step 4: calculate expectation,sigma for mmr_mean
	double E_mmr_mean=0;
	double sigma_mmr_mean=0;	
	for(int i=0;i<mmr_mean.size();i++){
		E_mmr_mean+=mmr_mean[i];
	}
	E_mmr_mean=E_mmr_mean/mmr_mean.size();

	for(int i=0;i<mmr_mean.size();i++){
		sigma_mmr_mean+=pow(mmr_mean[i]-E_mmr_mean,2);
	}
	sigma_mmr_mean=sqrt(sigma_mmr_mean/(mmr_mean.size()-1));

	// Step 5: detect outlier
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		double err=abs(mmr_mean[i]-E_mmr_mean);
		if(err>=3*sigma_mmr_mean){
			cloud->points[i].r=255;
			cloud->points[i].g=0;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileBinary("mmr.ply",*cloud);
}

// Radius, Central Limit Theorem
void CLT_Radius(pcl::PointCloud<PointType>::Ptr cloud)
{
	// Step 1: Init Parameters
	int K=32;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);

	// Step 2: calculate radius
	vector<double> r(cloud->points.size());
	vector<double> rmean(cloud->points.size());
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// Step 2-1: Define Parameters 
		vector<int> idx(K);
		vector<float> dist(K);
		pcl::PointCloud<PointType>::Ptr ctmp(new pcl::PointCloud<PointType>);

		// Step 2-2: Establish cloud temp
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		double dtmp=0;
		for(int j=0;j<dist.size();j++){
			dtmp+=dist[j];
		}
		r[i]=dtmp/K;
	}

	// Step 3: calculate sample mean
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		// step 3-1: find k-nearest neighbours
		vector<int> idx(K);
		vector<float> dist(K);
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		// step 3-2: calculate mean of radius
		double mtmp=0;
		for(int j=0;j<K;j++){
			mtmp+=r[idx[j]];
		}
		mtmp=mtmp/K;
		// step 3-3: store rmean
		rmean[i]=mtmp;
	}

	// Step 4: calculate expectation,sigma for mmr_mean
	double E_rmean=0;
	double sigma_rmean=0;	
	for(int i=0;i<rmean.size();i++){
		E_rmean+=rmean[i];
	}
	E_rmean=E_rmean/rmean.size();

	for(int i=0;i<rmean.size();i++){
		sigma_rmean+=pow(rmean[i]-E_rmean,2);
	}
	sigma_rmean=sqrt(sigma_rmean/(rmean.size()-1));

	// Step 5: detect outlier
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		double err=abs(rmean[i]-E_rmean);
		if(err>=3*sigma_rmean){
			cloud->points[i].r=0;
			cloud->points[i].g=255;
			cloud->points[i].b=0;
		}
	}
	pcl::io::savePLYFileBinary("R_CLT.ply",*cloud);
}

