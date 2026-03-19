#ifndef CONVEXHULL_HPP_
#define CONVEXHULL_HPP_

// A C++ program to find convex hull of a set of points.Refer
// http://www.geeksforgeeks.org/orientation-3-ordered-points/
// for explanation of orientation()

#include <Eigen/Core>
//#include <iostream>
#include <stack>
#include <vector>

class ConvexHull {
private:
	// A utility function to find next to top in a stack
	static Eigen::Vector2f nextToTop(std::stack<Eigen::Vector2f> &S);

	// A utility function to swap two points
	static void swap(Eigen::Vector2f &p1, Eigen::Vector2f &p2);

	// A utility function to return square of distance
	// between p1 and p2
	static float distSq(Eigen::Vector2f p1, Eigen::Vector2f p2);

	// To find orientation of ordered triplet (p, q, r).
	// The function returns following values
	// 0 --> p, q and r are colinear
	// 1 --> Clockwise
	// 2 --> Counterclockwise
	static int orientation(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r);

	// A function used by library function qsort() to sort an array of
	// points with respect to the first point
	static int compare(const void *vp1, const void *vp2);

	// A globle point needed for  sorting points with reference
	// to  the first point Used in compare function of qsort()

	static Eigen::Vector2f p0;

public:
	static float convexHull(std::vector<Eigen::Vector2f> points);

};


#endif /* CONVEXHULL_HPP_ */