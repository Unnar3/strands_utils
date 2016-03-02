#ifndef PRIMITIVE_EXTRACTOR_H
#define PRIMITIVE_EXTRACTOR_H

#include <pcl/point_cloud.h>
#include <pcl/octree/octree_impl.h>
#include <Eigen/Dense>
#include "primitive_params.h"
#include "primitive_octree.h"
#include "base_primitive.h"
#include "primitive_visualizer.h"

template <typename Point>
class primitive_extractor {
public:
    typedef Point point_type;
    typedef pcl::PointCloud<point_type> cloud_type;
    typedef typename cloud_type::Ptr cloud_ptr;
    typedef typename cloud_type::ConstPtr cloud_const_ptr;
    typedef primitive_octree<point_type> octree_type;
protected:
    cloud_ptr cloud; // pcl formats
    primitive_params params; // the parameters of the algorithm, from primitive_params.h
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals;
    Eigen::MatrixXd mpoints; // points of cloud
    Eigen::MatrixXd mnormals; // normals of point cloud
private:
    octree_type octree; // octree used for the entire cloud
    int tree_depth; // the levels of the octree
    Eigen::ArrayXi level_scores; // the sum of the number of primitives at each level of the octree
    //Eigen::VectorXd level_pdf;
    std::vector<base_primitive*> candidates; // candidates for new primitives
    int number_extracted; // number of primitives extracted, used for colors
    std::vector<octree_type> octrees; // the disjoint subsets as octrees
    std::vector<int> total_set_size; // the cumulative sum of number of points in the octrees

    std::vector<base_primitive*>& primitives; // primitive types used

    primitive_visualizer<point_type>* vis; // tool for visualizing the progression
    bool extract_single;


    void remove_distant_points(cloud_ptr new_cloud, double dist);
    void estimate_normals();
    int sample_level(int iteration);
    void get_points_at_level(std::vector<int>& inds, point_type& p, int level);
    double prob_candidate_not_found(double candidate_size,
                                    double candidates_evaluated,
                                    int points_required);
    void remove_points_from_cloud(base_primitive* p);
    void color_primitive(base_primitive* p);
    void add_new_primitive(base_primitive* primitive);
    void clear_primitives(std::vector<base_primitive *>& ps);
    void construct_octrees();
    base_primitive* max_inliers(double& maxmean, double& maxa, double& maxb,
                                std::vector<base_primitive*>& primitives);
    void overlapping_estimates(std::vector<base_primitive*>& primitives, base_primitive* best_candidate);
    double refine_inliers(std::vector<base_primitive *>& primitives);
public:
    void extract_single_param(bool single){
        extract_single = single;
    }
    void primitive_inlier_points(Eigen::MatrixXd& points, base_primitive* p);
    void extract(std::vector<base_primitive*>& extracted);
    pcl::PointCloud<pcl::Normal>::ConstPtr get_normals();
    cloud_const_ptr get_cloud();
    primitive_extractor(cloud_ptr cloud,
                        std::vector<base_primitive*>& primitives,
                        primitive_params params = primitive_params(),
                        primitive_visualizer<point_type>* vis = NULL);
    primitive_extractor(cloud_ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr cloud_normals,
                        std::vector<base_primitive*>& primitives,
                        primitive_params params = primitive_params(),
                        primitive_visualizer<point_type>* vis = NULL);
};

#include "primitive_extractor.hpp"
#endif // PRIMITIVE_EXTRACTOR_H
