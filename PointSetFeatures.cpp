#include "PointSetFeatures.h"
#include "RunPython.h"
double Feature::GetNormalAngle(pcl::PointCloud<PointType>::Ptr cloud)
{
    vector<Rrd> angle;
    Eigen::Vector3d v0= Eigen::Vector3d(cloud->points[0].x,cloud->points[0].y,cloud->points[0].z);
    for(int i=1;i<cloud->points.size()-2;i++){
        Eigen::Vector3d v1=Eigen::Vector3d(cloud->points[i].x,cloud->points[i].y,cloud->points[i].z);
        Eigen::Vector3d v2=Eigen::Vector3d(cloud->points[i+1].x,cloud->points[i+1].y,cloud->points[i+1].z);
        Eigen::Vector3d v3=Eigen::Vector3d(cloud->points[i+2].x,cloud->points[i+2].y,cloud->points[i+2].z);
        Eigen::Vector3d v01=v1-v0;
        Eigen::Vector3d v02=v2-v0;
        Eigen::Vector3d v03=v3-v0;
        Eigen::Vector3d n012=v01.cross(v02);
        Eigen::Vector3d n023=v02.cross(v03);
        double arc=acos(n012.normalized().dot(n023.normalized()));
        if(arc>M_PI/2.0)
            arc=M_PI-arc;

        angle.push_back(Rrd(i,arc));
    }
    sort(angle.begin(),angle.end(),[](Rrd& e1, Rrd& e2){ return e1.item1_<e2.item1_;});
    
    return angle[angle.size()-1].item1_;
}

void Feature::ComputeMahalanobisDistance(Eigen::MatrixXd v, Eigen::MatrixXd S)
{
	mean_.resize(S.rows(),S.cols());	
	cov_.resize(S.rows(),S.rows());
	cov_.setZero();
	mean_.setZero();

	// Step 01: Calculate Mean
	Eigen::MatrixXd dmean(S.rows(),1);
	dmean.setZero();
	for(int i=0;i<S.rows();i++){
		dmean(i)=S.row(i).mean();
	}
	for(int i=0;i<S.rows();i++){
		for(int j=0;j<S.cols();j++){
			mean_(i,j)=dmean(i);
			mean_(i,j)=dmean(i);
			mean_(i,j)=dmean(i);
		}
	}

	// Step 02: Calculate Covariance	
	Eigen::MatrixXd S_zero_mean=S-mean_;	
	for(int i=0;i<S.rows();i++){
		for(int j=0;j<S.rows();j++){			
			cov_(i,j)=S_zero_mean.row(i)*S_zero_mean.row(j).transpose();
			cov_(i,j)=cov_(i,j)/(S.cols()-1);			
		}
	}

	// Step 03: Mahanalbis Distance		
	Eigen::MatrixXd cov_inv=cov_.inverse();	
    Mdist_=(v-dmean.col(0)).transpose()*cov_inv*(v-dmean.col(0));
    Mdist_(0,0)=sqrt(Mdist_(0,0));
    // cout<<Mdist_<<endl;
}

double Feature::Poly33(pcl::PointCloud<PointType>::Ptr cloud)
{
    Eigen::MatrixXd A(cloud->points.size(),10);
    Eigen::MatrixXd ztmp(cloud->points.size(),1);
    for(int i=0;i<cloud->points.size();i++){
        ztmp(i,0)=cloud->points[i].z;
    }

	for(int i=0;i<cloud->points.size();i++){
		double datx=cloud->points[i].x;
		double daty=cloud->points[i].y;
		A(i,0)=1;
		A(i,1)=datx;
		A(i,2)=daty;
		A(i,3)=pow(datx,2);
		A(i,4)=datx*daty;
		A(i,5)=pow(daty,2);
		A(i,6)=pow(datx,3);
		A(i,7)=pow(datx,2)*daty;
		A(i,8)=datx*pow(daty,2);
		A(i,9)=pow(daty,3);
	}
	P_=(A.transpose()*A).inverse()*A.transpose()*ztmp;

    double xtmp=cloud->points[0].x;
    double ytmp=cloud->points[0].y;
    Eigen::MatrixXd dtmp(10,1);
    dtmp<< 1, xtmp, ytmp, pow(xtmp,2), xtmp*ytmp, pow(ytmp,2), pow(xtmp,3), pow(xtmp,2)*ytmp, xtmp*pow(ytmp,2), pow(ytmp,3);
    Eigen::MatrixXd ztmpMat=P_.transpose()*dtmp;

    double denominator= abs(ztmp(0,0) - ztmpMat(0,0));
    double partial_x=-(P_(1,0) + 2*P_(3,0)*xtmp + P_(4,0)*ytmp + 3*P_(6,0)*pow(ytmp,2) + 2*P_(7,0)*xtmp*ytmp + P_(8,0)*pow(ytmp,2));
    double partial_y=-(P_(2,0) + P_(4,0)*xtmp + 2*P_(5,0)*ytmp + P_(7,0)*pow(xtmp,2) + 2*P_(8,0)*xtmp*ytmp + 3*P_(9,0)*pow(ytmp,2));
    double numerator= sqrt(pow(partial_x,2)+pow(partial_y,2)+1);
    return denominator/numerator;
}

int Feature::KnnPlaneCounter(pcl::PointCloud<PointType>::Ptr cloud)
{
    vector<V3> arrows;
    for(int j=1;j<cloud->points.size();j++){
        if(j!=0){
            arrows.push_back(V3(cloud->points[j].x-cloud->points[0].x,
                                cloud->points[j].y-cloud->points[0].y,
                                cloud->points[j].z-cloud->points[0].z));
        }
    }
    EvalAndEvec vv(cloud);
    V3 ntmp(vv.eigenvector_[0].x,vv.eigenvector_[0].y,vv.eigenvector_[0].z);
    double count=0;
    for(int j=0;j<arrows.size();j++){
        double plen=Dot(arrows[j],ntmp)/ntmp.GetLength()/arrows[j].GetLength();
        if(abs(plen)<0.11)
            count++;
    }
    return count;
}

double Feature::KnnPlaneProjector(pcl::PointCloud<PointType>::Ptr cloud)
{
    vector<V3> arrows;
    for(int j=1;j<cloud->points.size();j++){
        if(j!=0){
            arrows.push_back(V3(cloud->points[j].x-cloud->points[0].x,
                                cloud->points[j].y-cloud->points[0].y,
                                cloud->points[j].z-cloud->points[0].z));
        }
    }
    EvalAndEvec vv(cloud);
    V3 ntmp(vv.eigenvector_[0].x,vv.eigenvector_[0].y,vv.eigenvector_[0].z);
    double Prj=0;
    for(int j=0;j<arrows.size();j++){
        Prj+=Dot(arrows[j],ntmp)/ntmp.GetLength();
    }
    return Prj;
}

PointSetFeatures::PointSetFeatures(pcl::PointCloud<PointType>::Ptr cloud)
{
    cloud_=cloud;
    cloud_trusted_=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    cloud_untrusted_=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
    rst_.Resize(cloud->points.size());
}

void PointSetFeatures::ApplyMinorEigenvalue(pcl::PointCloud<PointType>::Ptr cloud, int K)
{
    flag_MinorEigenvalue_=1;
    rst_MinorEigenvalue_.Resize(cloud->points.size());
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);
	
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		vector<int> idx(K);
		vector<float> dist(K);		
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);
		pcl::PointCloud<PointType>::Ptr ptmp_k(new pcl::PointCloud<PointType>);     
		for(int j=0;j<idx.size();j++){
			ptmp_k->points.push_back(cloud->points[idx[j]]);
		}
        EvalAndEvec vv(ptmp_k);
		rst_MinorEigenvalue_.records_[i].id_=i;
		rst_MinorEigenvalue_.records_[i].item1_=vv.eigenvalue_[0];
	}
    
}

void PointSetFeatures::ApplyEigenvalueRatio(pcl::PointCloud<PointType>::Ptr cloud, int K)
{    
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
	kdtree->setInputCloud(cloud);
	
	#pragma omp parallel for
	for(int i=0;i<cloud->points.size();i++){
		vector<int> idx(K);
		vector<float> dist(K);		
		kdtree->nearestKSearch(cloud->points[i], K, idx, dist);		
        pcl::PointCloud<PointType>::Ptr ptmp_k(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr ptmp_5(new pcl::PointCloud<PointType>);
		for(int j=0;j<idx.size();j++){
			ptmp_k->points.push_back(cloud->points[idx[j]]);
		}
        for(int j=0;j<5;j++){
			ptmp_5->points.push_back(cloud->points[idx[j]]);
		}

		EvalAndEvec vv;
        vv.GetEvalAndEvec(ptmp_5);
        vv.GetEvalAndEvec(ptmp_k);		
		rst_.records_[i].id_=i;
		rst_.records_[i].item1_=vv.eigenvalue_[1]/vv.eigenvalue_[2];
	}
}

void PointSetFeatures::ApplyQuadricSurfaceFitting(pcl::PointCloud<PointType>::Ptr cloud)
{
    int K=30;
    pcl::search::KdTree<PointType>::Ptr kdtree (new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);
        for(int j=0;j<idx.size();j++){
            ctmp->points.push_back(cloud->points[idx[j]]);
        }

        Feature f;
        rst_.records_[i].id_=i;
        rst_.records_[i].item1_=f.Poly33(ctmp);
    }
}

void PointSetFeatures::knnNormaAngle(pcl::PointCloud<PointType>::Ptr cloud)
{
    rst_.Resize(cloud->points.size());
    int K=7;    
    pcl::search::KdTree<PointType>::Ptr kdtree (new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){        
        vector<int> idx(K);
        vector<float> dist(K);
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);
        for(int j=0;j<idx.size();j++){
            ctmp->points.push_back(cloud->points[idx[j]]);
        }
        Feature fna;
        rst_.records_[i].id_=i;
        rst_.records_[i].item1_=fna.GetNormalAngle(ctmp);
    }
    ofstream fout("Result/0.csv");
    for(int i=0;i<rst_.Rows();i++){
        fout<<rst_.records_[i].item1_<<endl;
    }
    fout.close();

    for(int i=0;i<cloud->points.size();i++){
        if(rst_.records_[i].item1_>1.4){
            cloud->points[i].r=255;
            cloud->points[i].g=0;
            cloud->points[i].b=0;
        }
    }
    pcl::io::savePLYFileBinary("Result/knnNormaAngle.ply",*cloud);
}

void PointSetFeatures::ApplyPLOF(int K)
{
    PLOF_.resize(cloud_->points.size());
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud_);    
    #pragma omp parallel for
    for(int i=0;i<cloud_->points.size();i++){
        vector<int> idx(K+1);
        vector<float> dist(K+1);        
        kdtree->nearestKSearch(cloud_->points[i], K+1, idx, dist);
        double sum=0;
        for(int j=1;j<K+1;j++){
            sum+=rst_StandardizedEuclideanDistance_.records_[idx[j]].item1_;
        }
        sum/=K;
        PLOF_[i]=rst_StandardizedEuclideanDistance_.records_[i].item1_/sum-1.0f;
    }

    for(int i=0;i<cloud_->points.size();i++){
        rst_StandardizedEuclideanDistance_.records_[i].item1_=PLOF_[i];
    }
}

void PointSetFeatures::ApplynPLOF()
{    
    double nPLOF=0;
    for(int i=0;i<cloud_->points.size();i++){
        nPLOF+=PLOF_[i]*PLOF_[i];
    }
    nPLOF=sqrt(nPLOF/cloud_->points.size());
    nPLOF=1.0;

    for(int i=0;i<cloud_->points.size();i++){
        double value = PLOF_[i]/(nPLOF*sqrt(2.0));
        double dem = 1.0 + 0.278393 * value;
        dem += 0.230389 * value * value;
        dem += 0.000972 * value * value * value;
        dem += 0.078108 * value * value * value * value;
        rst_StandardizedEuclideanDistance_.records_[i].item1_ = std::max(0.0, 1.0 - 1.0 / dem);
    }
}

void PointSetFeatures::ApplyStandardizedEuclideanDistance(pcl::PointCloud<PointType>::Ptr cloud,int K)
{
    flag_StandardizedEuclideanDistance_=1;
    rst_StandardizedEuclideanDistance_.Resize(cloud->points.size());
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K+1);
        vector<float> dist(K+1);
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch(cloud->points[i], K+1, idx, dist);
        double sum=0;
        for(int j=1;j<K+1;j++){
            sum+=dist[j];
        }
        sum=sqrt(sum/K);
        rst_StandardizedEuclideanDistance_.records_[i].id_=i;
        rst_StandardizedEuclideanDistance_.records_[i].item1_=sum;
    }
    ApplyPLOF(K);
    ApplynPLOF();
}

void PointSetFeatures::ApplyMahalanobis(pcl::PointCloud<PointType>::Ptr cloud)
{   
    int K=25;
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);
        // Convert to Matrix    
        Eigen::MatrixXd v;
        Eigen::MatrixXd ctmp;
        ctmp.resize(3,idx.size()-1);
        v.resize(3,1);
        v<<cloud->points[i].x,cloud->points[i].y,cloud->points[i].z;
        for(int j=1;j<idx.size();j++){
            ctmp(0,j-1)=cloud->points[idx[j]].x;
            ctmp(1,j-1)=cloud->points[idx[j]].y;
            ctmp(2,j-1)=cloud->points[idx[j]].z;
        }
        Feature f;
        f.ComputeMahalanobisDistance(v,ctmp);
        rst_.records_[i].id_=i;
        rst_.records_[i].item1_=f.Mdist_(0,0);
    }
}

void PointSetFeatures::ApplyStandardDistance(pcl::PointCloud<PointType>::Ptr cloud)
{
    int K=32;
    pcl::search::KdTree<PointType>::Ptr kdtree (new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);
        // compute mean for local cloud set
        double x_bar,y_bar,z_bar;
        x_bar=y_bar=z_bar=0;
        for(int j=0;j<idx.size();j++){
            x_bar+=cloud->points[idx[j]].x;
            y_bar+=cloud->points[idx[j]].y;
            z_bar+=cloud->points[idx[j]].z;
        }
        x_bar=x_bar/K;
        y_bar=y_bar/K;
        z_bar=z_bar/K;

        // compute sd
        double sd=0;
        for(int j=0;j<idx.size();j++){
            sd+=pow(cloud->points[idx[j]].x-x_bar,2)+pow(cloud->points[idx[j]].y-y_bar,2)+pow(cloud->points[idx[j]].z-z_bar,2);
        }
        sd=sqrt(sd/K);
        rst_.records_[i].id_=i;
        rst_.records_[i].item1_=sd;
    }
}


void PointSetFeatures::Write(string path,pcl::PointCloud<PointType>::Ptr cloud)
{
    if(flag_MinorEigenvalue_==1 && flag_KnnPlane_==1){
        rst_MinorEigenvalue_.Normalize_Min_Max();
        rst_KnnPlane_cnt_.Normalize_Min_Max();
        for(int i=0;i<cloud_->points.size();i++){
            rst_.records_[i].item1_=rst_MinorEigenvalue_.records_[i].item1_*rst_KnnPlane_cnt_.records_[i].item1_;
        }
    }
    else if(flag_density_==1 && flag_KnnPlane_==1){
        // density outer
        // rst_density_.GetBoxplot(15.0);
        // rst_density_.GetBoxplot(5.0);
        // for(int i=0;i<rst_density_.inactive_idx_.size();i++){
        //     int itmp=rst_density_.inactive_idx_[i];
        //     cloud->points[itmp].r=255;
        //     cloud->points[itmp].g=0;
        //     cloud->points[itmp].b=0;
        // }

        // knn plane outer
        rst_KnnPlane_cnt_.GetBoxplot();
        for(int i=0;i<rst_KnnPlane_cnt_.inactive_idx_.size();i++){
            int itmp=rst_KnnPlane_cnt_.inactive_idx_[i];
            cloud->points[itmp].r=255;
            cloud->points[itmp].g=0;
            cloud->points[itmp].b=0;
        }

        pcl::io::savePLYFileBinary(path,*cloud);
    }
    else if(flag_MinorEigenvalue_==1){
        rst_MinorEigenvalue_.GetCorrespondingColor();
        for(int i=0;i<rst_MinorEigenvalue_.color_.size();i++){
            cloud->points[i].r=rst_MinorEigenvalue_.color_[i].r;
            cloud->points[i].g=rst_MinorEigenvalue_.color_[i].g;
            cloud->points[i].b=rst_MinorEigenvalue_.color_[i].b;
        }
        pcl::io::savePLYFileBinary(path,*cloud);
    }
    else if(flag_KnnPlane_==1){
        rst_KnnPlane_cnt_.GetCorrespondingColor();
        for(int i=0;i<rst_KnnPlane_cnt_.color_.size();i++){
            cloud->points[i].r=rst_KnnPlane_cnt_.color_[i].r;
            cloud->points[i].g=rst_KnnPlane_cnt_.color_[i].g;
            cloud->points[i].b=rst_KnnPlane_cnt_.color_[i].b;
        }

        // rst_KnnPlane_cnt_.GetBoxplot(25);
        pcl::io::savePLYFileBinary(path,*cloud);
    }
    else if(flag_KnnPlaneProjection_==1)
    {
        rst_KnnPlaneProjection_.Normalize_Min_Max();
        for(int i=0;i<cloud_->points.size();i++){
            rst_.records_[i].item1_=rst_KnnPlaneProjection_.records_[i].item1_;
        }
    }
    else if(flag_CentroidAndCentre_==1){        
        rst_CentroidAndCentre_.Normalize_Min_Max();
        rst_CentroidAndCentre_.GetBoxplot();
        rst_CentroidAndCentre_.EnableActive();

        // 
        for(int i=0;i<rst_CentroidAndCentre_.lower_inactive_idx_.size();i++){
            int itmp=rst_CentroidAndCentre_.lower_inactive_idx_[i];
            cloud->points[itmp].r=0;
            cloud->points[itmp].g=0;
            cloud->points[itmp].b=255;
        }

        for(int i=0;i<rst_CentroidAndCentre_.upper_inactive_idx_.size();i++){
            int itmp=rst_CentroidAndCentre_.upper_inactive_idx_[i];
            cloud->points[itmp].r=255;
            cloud->points[itmp].g=0;
            cloud->points[itmp].b=0;
        }

        rst_CentroidAndCentre_.GetMinimumAndMaximum();
        for(int i=0;i<rst_CentroidAndCentre_.active_idx_.size();i++){
            int itmp=rst_CentroidAndCentre_.active_idx_[i];
            V3 ctmp=get_color(rst_CentroidAndCentre_.min_,rst_CentroidAndCentre_.max_,rst_CentroidAndCentre_.records_[itmp].item1_);
            cloud->points[itmp].r=ctmp.r;
            cloud->points[itmp].g=ctmp.g;
            cloud->points[itmp].b=ctmp.b;
        }
        pcl::io::savePLYFileBinary(path,*cloud);
        rst_CentroidAndCentre_.DisableActive();
    }
    else if(flag_StandardizedEuclideanDistance_==1){       
        rst_StandardizedEuclideanDistance_.GetCorrespondingColor();
        for(int i=0;i<rst_StandardizedEuclideanDistance_.color_.size();i++){
            cloud->points[i].r=rst_StandardizedEuclideanDistance_.color_[i].r;
            cloud->points[i].g=rst_StandardizedEuclideanDistance_.color_[i].g;
            cloud->points[i].b=rst_StandardizedEuclideanDistance_.color_[i].b;
        }
        pcl::io::savePLYFileBinary(path,*cloud);
    }
    else if(flag_density_==1){
        rst_density_.GetCorrespondingColor();
        for(int i=0;i<rst_density_.color_.size();i++){
            cloud->points[i].r=rst_density_.color_[i].r;
            cloud->points[i].g=rst_density_.color_[i].g;
            cloud->points[i].b=rst_density_.color_[i].b;
        }
        pcl::io::savePLYFileBinary(path,*cloud);
    }    
}

void PointSetFeatures::ApplyKnnPlane(pcl::PointCloud<PointType>::Ptr cloud,int K)
{
    flag_KnnPlane_=1;
    rst_KnnPlane_cnt_.Resize(cloud->points.size());
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);

    #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        double mean_dist=0;
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);        
        for(int j=0;j<idx.size();j++){
            ctmp->points.push_back(cloud->points[idx[j]]);            
        }        

        Feature f;
        int cnt=f.KnnPlaneCounter(ctmp);
        rst_KnnPlane_cnt_.records_[i].id_=i;
        rst_KnnPlane_cnt_.records_[i].item1_=cnt;
    }
}

void PointSetFeatures::ApplyDensity(pcl::PointCloud<PointType>::Ptr cloud,double lamda)
{
    flag_density_=1;
    double cloud_mean_dist=ComputeMeanDistance(cloud);
    rst_density_.Resize(cloud->points.size());
    pcl::search::KdTree<PointType>::Ptr kdtree (new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);

    // Step 01: Get Feature (Density)
    #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx;
        vector<float> dist;
        kdtree->radiusSearch(cloud->points[i], lamda*cloud_mean_dist, idx, dist);    
        rst_density_.records_[i].id_=i;
        rst_density_.records_[i].item1_=1.0/idx.size();
    }

    // Step 02: Local Ratio
    int K=20;
    Table rst_density_backup;
    rst_density_backup.Resize(rst_density_.records_.size());    
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        kdtree->nearestKSearch(cloud->points[i], K, idx, dist);

        double sigma=0;
        for(int j=0;j<idx.size();j++){
            sigma+=rst_density_.records_[idx[j]].item1_;
        }
        sigma=sigma/idx.size();

        // Step 03: Store 
        rst_density_backup.records_[i].id_=i;
        rst_density_backup.records_[i].item1_=rst_density_.records_[i].item1_/sigma-1;
    }

    // Step 04: recovery
    for(int i=0;i<cloud->points.size();i++){
        rst_density_.records_[i].id_=rst_density_backup.records_[i].id_;
        rst_density_.records_[i].item1_=rst_density_backup.records_[i].item1_;
    }
}


void PointSetFeatures::ApplyKnnPlaneProjection(pcl::PointCloud<PointType>::Ptr cloud,int K)
{
    flag_KnnPlaneProjection_=1;
    rst_KnnPlaneProjection_.Resize(cloud->points.size());
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);

    #pragma omp parallel for
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);
        for(int j=0;j<idx.size();j++){
            ctmp->points.push_back(cloud->points[idx[j]]);
        }
        Feature f;
        rst_KnnPlaneProjection_.records_[i].id_=i;
        rst_KnnPlaneProjection_.records_[i].item1_=f.KnnPlaneProjector(ctmp);
    }
}

void PointSetFeatures::ApplyCentroidAndCentre(pcl::PointCloud<PointType>::Ptr cloud)
{
    flag_CentroidAndCentre_=1;
    rst_CentroidAndCentre_.Resize(cloud->points.size());
    int K=20;
    pcl::search::KdTree<PointType>::Ptr kdtree (new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud);
    for(int i=0;i<cloud->points.size();i++){
        vector<int> idx(K);
        vector<float> dist(K);
        pcl::PointCloud<PointType>::Ptr ctmp=pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        kdtree->nearestKSearch (cloud->points[i], K, idx, dist);
        V3 centroid;     
        for(int j=0;j<idx.size();j++){
            centroid.x+=cloud->points[idx[j]].x;
            centroid.y+=cloud->points[idx[j]].y;
            centroid.z+=cloud->points[idx[j]].z;
        }
        centroid.x=centroid.x/idx.size();
        centroid.y=centroid.y/idx.size();
        centroid.z=centroid.z/idx.size();

        V3 centre;
        centre.x=cloud->points[i].x;
        centre.y=cloud->points[i].y;
        centre.z=cloud->points[i].z;

        double dist_tmp=Distance(centroid,centre);
        rst_CentroidAndCentre_.records_[i].id_=i;
        rst_CentroidAndCentre_.records_[i].item1_=dist_tmp;
    }
}