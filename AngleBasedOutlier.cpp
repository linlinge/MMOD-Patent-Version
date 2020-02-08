#include "AngleBasedOutlier.h"
void AngleBasedOutlier::PointSetEntropy(pcl::PointCloud<PointType>::Ptr cloud)
{
    // Step 01: Define
    cloud_=cloud;
    int K=10>cloud_->points.size() ? cloud_->points.size()*0.6:10;
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud_);

    // Step 02: Calculate Entropy
    for(int i=0;i<cloud_->points.size();i++){
        AngleEntropy epy;
        epy.Init(9,10);
        // kNN neigbours
        pcl::PointCloud<PointType>::Ptr ptmp(new pcl::PointCloud<PointType>);
        vector<int> idx;
        vector<float> dist;
        kdtree->nearestKSearch(cloud_->points[i], K, idx, dist);
        for(int j=0;j<idx.size();j++){
            ptmp->points.push_back(cloud_->points[idx[j]]);
        }

        // ptmp
        for(int j=1;j<ptmp->points.size();j++){            
            epy.Insert(cloud_->points[idx[j]].x-cloud_->points[idx[0]].x,
                       cloud_->points[idx[j]].y-cloud_->points[idx[0]].y,
                       cloud_->points[idx[j]].z-cloud_->points[idx[0]].z);            
        }

        double current_entropy=epy.GetEntropy();
        point_set_entropy_.push_back(Rrd(i,current_entropy));
    }

    // sort entropy
    sort(point_set_entropy_.begin(),point_set_entropy_.end(),[](Rrd& e1, Rrd& e2){ return e1.item1_<e2.item1_;});
    // int quantile01=point_set_entropy_.size()*0.9;
    for(int i=0;i<point_set_entropy_.size();i++){
        int itmp=point_set_entropy_[i].id_;
        // if(point_set_entropy_[i].item1_<0.44 && point_set_entropy_[i].item1_>0.43){
        if(point_set_entropy_[i].item1_<0.41){
            cloud_->points[itmp].r=255;
            cloud_->points[itmp].g=0;
            cloud_->points[itmp].b=0;
        }
        cout<<point_set_entropy_[i].item1_<<endl;
    }
    // pcl::io::savePLYFileBinary("Result/entropy.ply",*cloud_);
}

void AngleBasedOutlier::PointSetCCD(pcl::PointCloud<PointType>::Ptr cloud)
{
     // Step 01: Define
    cloud_=cloud;
    int K=10>cloud_->points.size() ? cloud_->points.size()*0.6:10;
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(cloud_);
    vector<Rrd> ccd;    

    // Step 02: Calculate Entropy
    for(int i=0;i<cloud_->points.size();i++){
        // Step 02-1: kNN neigbours
        pcl::PointCloud<PointType>::Ptr ptmp(new pcl::PointCloud<PointType>);
        vector<int> idx;
        vector<float> dist;
        kdtree->nearestKSearch(cloud_->points[i], K, idx, dist);
        for(int j=0;j<idx.size();j++){
            ptmp->points.push_back(cloud_->points[idx[j]]);
        }

        // Step 02-2: compute centroid
        Eigen::Vector4f centroid;
        Eigen::Matrix3f covariance;
        pcl::compute3DCentroid(*ptmp, centroid);

        // Step 02-3: Calculate Distance of Centroid and Centre
        double dtmp=sqrt(pow(cloud_->points[i].x-centroid[0],2)+pow(cloud_->points[i].y-centroid[1],2)+pow(cloud_->points[i].z-centroid[2],2));
        // cout<<ccd<<endl;
        ccd.push_back(Rrd(i,dtmp));
    }
    // sort ccd
    sort(ccd.begin(),ccd.end(),[](Rrd& e1, Rrd& e2){ return e1.item1_<e2.item1_;});
    for(int i=0;i<ccd.size();i++)
    {
        // cout<<ccd[i].item1_<<endl;
        if(ccd[i].item1_>0.0009){
            int itmp=ccd[i].id_;
            cloud_->points[itmp].r=255;
            cloud_->points[itmp].g=0;
            cloud_->points[itmp].b=0;
        }
    }
    // pcl::io::savePLYFileBinary("Result/PointSetCCD.ply",*cloud_);
}