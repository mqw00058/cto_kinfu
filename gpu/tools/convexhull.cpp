#include "convexhull.h"

Eigen::Vector2f ConvexHull::nextToTop(std::stack<Eigen::Vector2f> &S)
{
	Eigen::Vector2f p = S.top();
	S.pop();
	Eigen::Vector2f res = S.top();
	S.push(p);
	return res;
}

// A utility function to swap two points
void ConvexHull::swap(Eigen::Vector2f &p1, Eigen::Vector2f &p2)
{
	Eigen::Vector2f temp = p1;
	p1 = p2;
	p2 = temp;
}

// A utility function to return square of distance
// between p1 and p2
float ConvexHull::distSq(Eigen::Vector2f p1, Eigen::Vector2f p2)
{
	return (p1.x() - p2.x())*(p1.x() - p2.x()) +
		(p1.y() - p2.y())*(p1.y() - p2.y());
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int ConvexHull::orientation(Eigen::Vector2f p, Eigen::Vector2f q, Eigen::Vector2f r)
{
	float val = (q.y() - p.y()) * (r.x() - q.x()) -
		(q.x() - p.x()) * (r.y() - q.y());

	if (fabs(val) < 0.001) return 0;  // colinear
	return (val > 0.0) ? 1 : 2; // clock or counterclock wise
}

// A function used by library function qsort() to sort an array of
// points with respect to the first point
int ConvexHull::compare(const void *vp1, const void *vp2)
{
	Eigen::Vector2f *p1 = (Eigen::Vector2f *)vp1;
	Eigen::Vector2f *p2 = (Eigen::Vector2f *)vp2;

	// Find orientation
	int o = orientation(p0, *p1, *p2);
	if (o == 0)
		return (distSq(p0, *p2) >= distSq(p0, *p1)) ? -1 : 1;

	return (o == 2) ? -1 : 1;
}

float ConvexHull::convexHull(std::vector<Eigen::Vector2f> points)
{
	// Find the bottommost point
	int n = points.size();
	float ymin = points[0].y();
	int min = 0;
	for (int i = 1; i < points.size(); i++)
	{
		float y = points[i].y();

		// Pick the bottom-most or chose the left
		// most point in case of tie
		if ((y < ymin) || (ymin == y &&
			points[i].x() < points[min].x()))
			ymin = points[i].y(), min = i;
	}

	// Place the bottom-most point at first position
	swap(points[0], points[min]);

	// Sort n-1 points with respect to the first point.
	// A point p1 comes before p2 in sorted ouput if p2
	// has larger polar angle (in counterclockwise
	// direction) than p1
	p0 = points[0];
	qsort(&points[1], n - 1, sizeof(Eigen::Vector2f), ConvexHull::compare);

	// If two or more points make same angle with p0,
	// Remove all but the one that is farthest from p0
	// Remember that, in above sorting, our criteria was
	// to keep the farthest point at the end when more than
	// one points have same angle.
	int m = 1; // Initialize size of modified array
	for (int i = 1; i<n; i++)
	{
		// Keep removing i while angle of i and i+1 is same
		// with respect to p0
		while (i < n - 1 && orientation(p0, points[i],
			points[i + 1]) == 0)
			i++;


		points[m] = points[i];
		m++;  // Update size of modified array
	}

	// If modified array of points has less than 3 points,
	// convex hull is not possible
	if (m < 3) return 0.0;

	// Create an empty stack and push first three points
	// to it.
	std::stack<Eigen::Vector2f> S;
	S.push(points[0]);
	S.push(points[1]);
	S.push(points[2]);

	// Process remaining n-3 points
	for (int i = 3; i < m; i++)
	{
		// Keep removing top while the angle formed by
		// points next-to-top, top, and points[i] makes
		// a non-left turn
		while (orientation(nextToTop(S), S.top(), points[i]) != 2)
			S.pop();
		S.push(points[i]);
	}

	// Now stack has the output points, print contents of stack
	float length = 0.0;
	Eigen::Vector2f p1, p2, ps;
	ps = p1 = S.top();
	S.pop();
	while (!S.empty())
	{
		p2 = S.top();
		S.pop();
		length += sqrt(distSq(p1, p2));
		p1 = p2;
	}
	length += sqrt(distSq(p2, ps));

	//while (!S.empty())
	//{
	//	Eigen::Vector2f p = S.top();
	//	cout << "(" << p.x << ", " << p.y << ")" << endl;
	//	S.pop();
	//}

	return length;
}

Eigen::Vector2f ConvexHull::p0;