#include "primitive_extractor.h"

#include <iomanip>
#include <time.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/passthrough.h>

#define PRINTOUTS false

using namespace Eigen;

template <typename Point>
primitive_extractor<Point>::primitive_extractor(cloud_ptr new_cloud,
                                                std::vector<base_primitive*>& primitives,
                                                primitive_params params, primitive_visualizer<point_type> *vis) :
    cloud(new cloud_type()), cloud_normals(new pcl::PointCloud<pcl::Normal>),
    octree(params.octree_res), primitives(primitives), params(params), vis(vis)
{
    // setup parameters
    base_primitive::number_disjoint_subsets = params.number_disjoint_subsets;
    base_primitive::min_inliers = params.inlier_min;
    base_primitive::connectedness_res = params.connectedness_res;
    extract_single = false;


    // seed the random number generator
    srand(time(NULL));

    // create the point cloud from the points that are close enough
    remove_distant_points(new_cloud, params.distance_threshold);

    // initialize clouds and matrices
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();
    tree_depth = octree.getTreeDepth() + 1;
    level_scores.resize(tree_depth);
    level_scores.setZero();
    construct_octrees();

    if (PRINTOUTS) {
        std::cout << "Octree constructed, tree depth: " << tree_depth << std::endl;
    }

    // estimate normals for all points
    estimate_normals();

    if (PRINTOUTS) {
        std::cout << "Normals extracted..." << std::endl;
    }

    mpoints.resize(3, cloud->size());
    mnormals.resize(3, cloud->size());

    Vector3f point;
    for (size_t i = 0; i < cloud->size(); ++i) {
        point = cloud->points[i].getVector3fMap();
        mpoints.col(i) = point.cast<double>();
        mnormals.col(i) = cloud_normals->points[i].getNormalVector3fMap().cast<double>();
        mnormals.col(i).normalize();
    }
}

template <typename Point>
primitive_extractor<Point>::primitive_extractor(cloud_ptr new_cloud, pcl::PointCloud<pcl::Normal>::Ptr cloud_normals,
                                                std::vector<base_primitive*>& primitives,
                                                primitive_params params, primitive_visualizer<point_type>* vis) :
    cloud(new cloud_type()), cloud_normals(new pcl::PointCloud<pcl::Normal>(*cloud_normals)),
    octree(params.octree_res), primitives(primitives), params(params), vis(vis)
{
    // setup parameters
    base_primitive::number_disjoint_subsets = params.number_disjoint_subsets;
    base_primitive::min_inliers = params.inlier_min;
    base_primitive::connectedness_res = params.connectedness_res;

    // seed the random number generator
    srand(time(NULL));

    // create the point cloud from the points that are close enough
    remove_distant_points(new_cloud, params.distance_threshold);

    // initialize clouds and matrices
    octree.setInputCloud(cloud);
    octree.addPointsFromInputCloud();
    tree_depth = octree.getTreeDepth() + 1;
    level_scores.resize(tree_depth);
    level_scores.setZero();
    construct_octrees();

    if (PRINTOUTS) {
        std::cout << "Octree constructed, tree depth: " << tree_depth << std::endl;
    }

    mpoints.resize(3, cloud->size());
    mnormals.resize(3, cloud->size());

    Vector3f point;
    for (size_t i = 0; i < cloud->size(); ++i) {
        point = cloud->points[i].getVector3fMap();
        mpoints.col(i) = point.cast<double>();
        mnormals.col(i) = cloud_normals->points[i].getNormalVector3fMap().cast<double>();
        mnormals.col(i).normalize();
    }
}

// setup the disjoint subsets
template <typename Point>
void primitive_extractor<Point>::construct_octrees()
{
    // randomize, setup the disjoint point sets
    std::vector<int> inds;
    inds.resize(cloud->size());
    for (size_t i = 0; i < cloud->size(); ++i) {
        inds[i] = i;
    }
    std::random_shuffle(inds.begin(), inds.end());

    octrees.resize(params.number_disjoint_subsets, octree_type(params.octree_res)); // could probably just pass an int?
    total_set_size.resize(params.number_disjoint_subsets);
    int sum = 0;

    // put them in the octrees
    int step = int(cloud->size())/params.number_disjoint_subsets; // assuming A >> B
    for (int i = 0; i < params.number_disjoint_subsets; ++i) {
        std::vector<int>::iterator start = inds.begin() + i*step;
        std::vector<int>::iterator end = inds.begin() + (i+1)*step;
        if (i == params.number_disjoint_subsets - 1) {
            end = inds.end();
        }
        octrees[i].setInputCloud(cloud, typename octree_type::IndicesConstPtr(new std::vector<int>(start, end)));
        octrees[i].addPointsFromInputCloud();
        sum += octrees[i].size();
        total_set_size[i] = sum;
    }
}

// for displaying the normals during the extraction
template <typename Point>
pcl::PointCloud<pcl::Normal>::ConstPtr primitive_extractor<Point>::get_normals()
{
    return cloud_normals;
}

// for displaying the cloud during the extraction
template <typename Point>
typename primitive_extractor<Point>::cloud_const_ptr primitive_extractor<Point>::get_cloud()
{
    return cloud;
}

// TODO: re-write this for cameras that are not in (0, 0, 0)
template <typename Point>
void primitive_extractor<Point>::remove_distant_points(cloud_ptr new_cloud, double dist)
{
    // if 0 is passed in as parameter, don't do anything
    if (dist == 0) {
        cloud->insert(cloud->end(), new_cloud->begin(), new_cloud->end());
        return;
    }
    // filter out points that are far from the camera and thus will contain too much noise
    pcl::PassThrough<point_type> pass;
    pass.setInputCloud(new_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, dist);
    pass.filter(*cloud);
}

// standard PCL normal extraction
template <typename Point>
void primitive_extractor<Point>::estimate_normals()
{
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimationOMP<point_type, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    typedef pcl::search::KdTree<point_type> kd_tree_type;
    typedef typename kd_tree_type::Ptr kd_tree_type_ptr;
    kd_tree_type_ptr tree(new kd_tree_type());
    //pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB>::Ptr ptr(octree);
    ne.setSearchMethod(tree);

    // Use all neighbors in a sphere of radius normal_radius m
    ne.setRadiusSearch(params.normal_neigbourhood);

    // Compute the features
    ne.compute(*cloud_normals);

    // cloud_normals->points.size () should have the same size as the input cloud->points.size ()*
}

// main function, extracts the primitives
template <typename Point>
void primitive_extractor<Point>::extract(std::vector<base_primitive*>& extracted)
{
    number_extracted = 0; // set numbers extracted to zero again
    extracted.clear(); // extracted primitives

    // loop through all types and find maximum points required to construct shape
    int min_set = 1;
    for (base_primitive* p : primitives) {
        if (p->points_required() > min_set) {
            min_set = p->points_required();
        }
    }

    int n = cloud->size();
    double candidates_evaluated = 0.0;

    double prob_not_found = 1.0;
    std::vector<int> inds;
    int iteration = 0;
    int find_valid_tries = 0;
    do {
        // pick one point from entire cloud
        int ind = rand() % n; // change to work for clouds > RAND_MAX
        if (isnan(cloud->points[ind].x) || isnan(cloud->points[ind].y) || isnan(cloud->points[ind].z)) {
            continue;
        }
        if (find_valid_tries > 10000) {
            std::cout << "Can't find a valid shape, done..." << std::endl;
            break;
        }

        // random tree depth, sample more points from that depth at point
        int level = sample_level(iteration);

        // sample enough points to extract all shapes
        MatrixXd points(3, min_set);
        MatrixXd normals(3, min_set);
        points.col(0) = mpoints.col(ind);
        normals.col(0) = mnormals.col(ind);

        // get points in same cell as first point at sampled depth
        inds.clear();
        get_points_at_level(inds, cloud->points[ind], level);
        if (inds.empty()) {
            continue;
        }

        // no need to check uniqueness, will not work well anyways
        for (int i = 1; i < min_set; ++i) {
            int ind = inds[rand() % inds.size()];
            points.col(i) = mpoints.col(ind);
            normals.col(i) = mnormals.col(ind);
        }

        // iterate over possible shapes, evaluate them to see if points good -> count inliers
        for (base_primitive* p : primitives) {
            base_primitive* c = p->instantiate();
            if (c->construct(points, normals, params.inlier_threshold, params.angle_threshold)) {
                c->refine_inliers(octrees, mpoints, mnormals, params.inlier_threshold, params.angle_threshold);
                if (c->supporting_inds.empty()) {
                    delete c;
                    continue;
                }
                candidates.push_back(c);
                level_scores(level) += c->supporting_inds.size(); // TODO: this must be expected value instead
                // should this get updated when intervals are refined? should be easy
            }
            else {
                delete c;
            }
        }
        candidates_evaluated += 1.0;

        // no candidates -> can't do anything
        if (candidates.size() == 0) {
            ++iteration;
            ++find_valid_tries;
            continue;
        }
        find_valid_tries = 0;

        int rounds = 0;
        std::vector<base_primitive*> best_candidates = candidates;
        double best_val;
        base_primitive* last_candidate = NULL;
        // refine confidence of inliers until we find the best non-overlapping
        // then see if it is the best again, among all candidates
        // if they are not the same, redo the whole scheme until they match up
        do {
            best_val = refine_inliers(best_candidates);
            if (best_candidates.size() == 1 && best_candidates[0] != last_candidate) {
                best_candidates[0]->final_inliers(octree, mpoints, mnormals, params.inlier_threshold, params.angle_threshold);
                last_candidate = best_candidates[0];
                best_candidates = candidates;
            }
            ++rounds;
        }
        while (best_candidates.size() > 1);

        base_primitive* best_candidate = best_candidates[0];
        if (PRINTOUTS) {
            std::cout << "Candidates: " << candidates.size() << std::endl;
            std::cout << "Extracted: " << extracted.size() << std::endl;
            std::cout << "Points left: " << octree.size() << std::endl;
            std::cout << "Best val:" << best_val << std::endl;
            std::cout << "Best cand prob: " << std::setprecision(10) << prob_candidate_not_found(double(best_val), candidates_evaluated,
                                                                        best_candidate->points_required()) << std::endl;
            std::cout << "Rounds: " << rounds << std::endl;
        }

        // if no better candidate can be found with P > 1 - add_threshold -> add to extracted, remove overlapping from candidates
        if (prob_candidate_not_found(best_val, candidates_evaluated, min_set) < params.add_threshold) {
            // make sure that we score the primitive against all the points (no confidence intervals)
            if (best_candidate->refinement_level() < params.number_disjoint_subsets) {
                best_candidate->final_inliers(octree, mpoints, mnormals, params.inlier_threshold, params.angle_threshold);
            }
            extracted.push_back(best_candidate);
            std::vector<base_primitive*> keep_candidates; // candidate to keep
            keep_candidates.reserve(candidates.size());
            for (base_primitive* p : candidates) {
                if (p == best_candidate) {
                    // do nothing
                }
                else if (p->are_contained(best_candidate->sorted_inliers())) {
                    // remove candidate
                    candidates_evaluated *= pow(1 - p->inliers_mean_estimate(octree.size(), total_set_size)/double(octree.size()), 3.0);
                    delete p;
                }
                else {
                    // keep in new vector
                    keep_candidates.push_back(p);
                }
            }
            // here we remove the points in the octree that are contained in best_candidate
            add_new_primitive(best_candidate);
            candidates.swap(keep_candidates);
            keep_candidates.clear();

            std::cout << "Extracted a primitive of size: " << best_candidate->supporting_inds.size() << std::endl;
            std::cout << "Octree depth level scores: " << level_scores.transpose() << std::endl;

            if(extract_single) break;
        }

        prob_not_found = prob_candidate_not_found(params.min_shape, candidates_evaluated, min_set);
        if (std::isinf(prob_not_found)) {
            clear_primitives(extracted);
            break;
        }

        if (PRINTOUTS) {
            std::cout << "Prob min cand not found: " << prob_not_found << std::endl;
        }
        ++iteration;
    }
    while (prob_not_found > params.add_threshold && octree.size() > params.min_shape);

    // min_set because that will be the most unlikely shape
    clear_primitives(candidates);

    // compute the sizes of the extracted primitives
    for (base_primitive* p : extracted) {
        p->compute_shape_size(mpoints);
    }
}

// finds the primitive with the highest expected inliers
template <typename Point>
base_primitive* primitive_extractor<Point>::max_inliers(double& maxmean, double& maxa, double& maxb,
                                                 std::vector<base_primitive*>& primitives)
{
    base_primitive* best_candidate = NULL;
    maxmean = -INFINITY;
    double mean, a, b;
    for (base_primitive* p : primitives) {
        p->inliers_estimate(mean, a, b, octree.size(), total_set_size);
        if (mean > maxmean) {
            maxmean = mean;
            maxa = a;
            maxb = b;
            best_candidate = p;
        }
    }
    return best_candidate;
}

// finds all prmimitives with inlier confidence interval overlapping the estimate of best_candidate
template <typename Point>
void primitive_extractor<Point>::overlapping_estimates(std::vector<base_primitive*>& primitives, base_primitive* best_candidate)
{
    double maxmean, maxa, maxb;
    best_candidate->inliers_estimate(maxmean, maxa, maxb, octree.size(), total_set_size);
    std::vector<base_primitive*> temp;
    temp.push_back(best_candidate);
    double mean, a, b;
    for (base_primitive* p : primitives) {
        if (p == best_candidate) {
            continue;
        }
        p->inliers_estimate(mean, a, b, octree.size(), total_set_size);
        // check if confidence intervals overlap
        if ((a < maxa && maxa < b) || (a < maxb && maxb < b) ||
                (maxa < a && a < maxb) || (maxa < b && b < maxb)) {
            temp.push_back(p);
        }
    }
    primitives.swap(temp);
}

template <typename Point>
double primitive_extractor<Point>::refine_inliers(std::vector<base_primitive*>& primitives)
{
    // find the candidate with the most expected inliers
    double maxmean, maxa, maxb;
    base_primitive* best_candidate = max_inliers(maxmean, maxa, maxb, primitives);
    int max_refinement = best_candidate->refinement_level();

    // find all candidates with overlapping bounds, discard the rest
    overlapping_estimates(primitives, best_candidate);

    // refine if there is more than one found
    if (primitives.size() == 1) {
        return maxmean;
    }
    for (base_primitive* p : primitives) {
        if (p->refinement_level() <= max_refinement) {
            p->refine_inliers(octrees, mpoints, mnormals, params.inlier_threshold, params.angle_threshold);
        }
    }

    return maxmean;
}

// remove all primitives from vector
template <typename Point>
void primitive_extractor<Point>::clear_primitives(std::vector<base_primitive*>& ps)
{
    for (base_primitive* p : ps) {
        delete p;
    }
    ps.clear();
}

// new primitive added to extracted, perform necessary operations on octrees etc.
template <typename Point>
void primitive_extractor<Point>::add_new_primitive(base_primitive* primitive)
{
    if (vis != NULL) {
        vis->lock();
        primitive->draw(vis->viewer);
        vis->unlock();
    }
    remove_points_from_cloud(primitive);
}

// remove contained points from the octrees
template <typename Point>
void primitive_extractor<Point>::remove_points_from_cloud(base_primitive* p)
{
    // remove points and compute the new total_set_size,
    // the cumulative sum of number of points in octrees,
    // needed when computing the confidence intervals of primitives
    octree.remove_points(p->supporting_inds); // remove points in octree that are part of shape
    int sum = 0;
    for (int i = 0; i < params.number_disjoint_subsets; ++i) {
        octrees[i].remove_points(p->supporting_inds);
        sum += octrees[i].size();
        total_set_size[i] = sum;
    }

    // visualize the extracted primitive
    if (vis != NULL) {
        vis->lock();
    }

    color_primitive(p);

    if (vis != NULL) {
        vis->cloud_changed = true;
        vis->unlock();
    }
}

// don't do anything if the points don't have color
template <typename Point>
void primitive_extractor<Point>::color_primitive(base_primitive* p)
{

}

// put color on the points
template <>
void primitive_extractor<pcl::PointXYZRGB>::color_primitive(base_primitive* p)
{
    int colormap[6][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 0, 255}, {255, 255, 0}, {64, 224, 208}};
    int r = number_extracted % 6;
    p->red = colormap[r][0];
    p->green = colormap[r][1];
    p->blue = colormap[r][2];
    ++number_extracted;

    for (const int& i : p->supporting_inds) {
        cloud->points[i].r = p->red;
        cloud->points[i].g = p->green;
        cloud->points[i].b = p->blue;
    }
}

// sample an octree level from the learnt distribution
template <typename Point>
int primitive_extractor<Point>::sample_level(int iteration)
{
    ArrayXd pdf;
    if (level_scores(0) < 200000) {
        pdf.resize(tree_depth);
        //return rand() % tree_depth; // sample with prob 1/d instead
        for (int j = 0; j < tree_depth; ++j) {
            pdf(j) = 1.0;//(1.0 + double(j));
        }
        pdf /= pdf.sum();
    }
    else {
        // rejection sampling approach
        double x = 0.9;
        double weight = double(level_scores.sum());
        pdf = x/weight*level_scores.cast<double>() + (1-x)*1/double(tree_depth);
        if (PRINTOUTS) {
            std::cout << "Pdf: " << pdf.transpose() << std::endl;
            std::cout << "Level scores: " << level_scores.transpose() << std::endl;
        }
    }
    double maxval = pdf.maxCoeff();
    int i;
    double d;
    while (true) {
        i = rand() % tree_depth;
        d = double(rand())/double(RAND_MAX);
        if (pdf[i] > d*maxval) {
            return i;
        }
    }
}

// get all points from a node containing x at a certain level of the octree
template <typename Point>
void primitive_extractor<Point>::get_points_at_level(std::vector<int>& inds, point_type& p, int level)
{
    octree.find_points_at_depth(inds, p, level);
}

// compute the probability of not having found a shape with candidate_size points
template <typename Point>
double primitive_extractor<Point>::prob_candidate_not_found(double candidate_size,
                                                            double candidates_evaluated,
                                                            int points_required)
{
    double intpart = octree.size()*tree_depth*(1 << points_required);
    return pow(1.0f - candidate_size/intpart, candidates_evaluated);
}

// get back the inlier points of an extracted primitive p
template <typename Point>
void primitive_extractor<Point>::primitive_inlier_points(MatrixXd& points, base_primitive* p)
{
    unsigned sz = p->supporting_inds.size();
    points.resize(3, sz);
    for (size_t i = 0; i < sz; ++i) {
        points.col(i) = mpoints.col(p->supporting_inds[i]);
    }
}
