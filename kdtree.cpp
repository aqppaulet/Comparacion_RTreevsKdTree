// kdtree.cpp (incluido por main.cpp) - NO compilar aparte

#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>

class KDTree {
private:
    struct Node {
        Point point;
        int split_dim;
        Node* left;
        Node* right;

        Node(const Point& p, int dim) : point(p), split_dim(dim), left(nullptr), right(nullptr) {}
        ~Node() { delete left; delete right; }
    };

    struct KNNResult {
        double distance_sq;
        Point point;

        // max-heap: el peor vecino queda arriba
        bool operator<(const KNNResult& other) const {
            return distance_sq < other.distance_sq;
        }
    };

    Node* root;
    int D;
    long long node_count;

    Node* build(std::vector<Point>& points, int depth) {
        if (points.empty()) return nullptr;

        int dim = depth % D;

        std::sort(points.begin(), points.end(), [dim](const Point& a, const Point& b) {
            return a.get(dim) < b.get(dim);
        });

        size_t mid = points.size() / 2;
        Point median = points[mid];

        Node* node = new Node(median, dim);
        node_count++;

        std::vector<Point> leftPoints(points.begin(), points.begin() + mid);
        std::vector<Point> rightPoints(points.begin() + mid + 1, points.end());

        node->left = build(leftPoints, depth + 1);
        node->right = build(rightPoints, depth + 1);

        return node;
    }

    void knn_recursive(Node* node, const Point& query, int k,
                       std::priority_queue<KNNResult>& pq) const
    {
        if (!node) return;

        double dist_sq = node->point.distanceSq(query);

        if ((int)pq.size() < k) {
            pq.push({dist_sq, node->point});
        } else if (dist_sq < pq.top().distance_sq) {
            pq.pop();
            pq.push({dist_sq, node->point});
        }

        int dim = node->split_dim;
        double diff = query.get(dim) - node->point.get(dim);

        Node* near_child = (diff < 0) ? node->left : node->right;
        Node* far_child  = (diff < 0) ? node->right : node->left;

        knn_recursive(near_child, query, k, pq);

        double radius_sq = ((int)pq.size() < k) ? std::numeric_limits<double>::max() : pq.top().distance_sq;
        double plane_dist_sq = diff * diff;

        if (plane_dist_sq < radius_sq) {
            knn_recursive(far_child, query, k, pq);
        }
    }

public:
    KDTree(int dimensions = INITIAL_DIM) : root(nullptr), D(dimensions), node_count(0) {}

    ~KDTree() { delete root; }

    void insert_batch(std::vector<Point>& points) {
        auto start = std::chrono::high_resolution_clock::now();
        root = build(points, 0);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << "Tiempo de Inserción (Batch): "
                  << std::fixed << std::setprecision(6)
                  << duration.count() << " segundos\n";
    }

    std::vector<Point> kNearestNeighbors(const Point& query, int k) const {
        if (!root || k <= 0) return {};

        std::priority_queue<KNNResult> pq;

        auto start = std::chrono::high_resolution_clock::now();
        knn_recursive(root, query, k, pq);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << "Tiempo de Búsqueda (KNN, k=" << k << "): "
                  << std::fixed << std::setprecision(6)
                  << duration.count() << " segundos\n";

        std::vector<Point> results;
        while (!pq.empty()) {
            results.push_back(pq.top().point);
            pq.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }

    long long getNodeCount() const { return node_count; }
    int getDimensions() const { return D; }
};
