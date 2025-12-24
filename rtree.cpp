// rtree.cpp (incluido por main.cpp) - NO compilar aparte
// R-Tree con KNN exacto (best-first) usando minDist a MBR.

#include <vector>
#include <queue>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

class RTree {
private:
    static constexpr int MAX_ENTRIES = 16;
    static constexpr int MIN_ENTRIES = 8;
    static constexpr double EPS = 1e-9;

    struct Rect {
        std::array<double, MAX_DIM> mn{};
        std::array<double, MAX_DIM> mx{};
        int D = INITIAL_DIM;
        explicit Rect(int d = INITIAL_DIM) : D(d) {
            mn.fill(0.0);
            mx.fill(0.0);
        }
    };

    struct Node;

    struct Entry {
        Rect mbr;
        Node* child;   // interno
        Point point;   // hoja
        bool has_point; // true si es hoja

        Entry(const Rect& r, Node* c, int D) : mbr(r), child(c), point(D), has_point(false) {}
        Entry(const Rect& r, const Point& p) : mbr(r), child(nullptr), point(p), has_point(true) {}
    };

    struct Node {
        bool leaf;
        int D;
        Node* parent;
        std::vector<Entry> entries;

        explicit Node(bool is_leaf, int d) : leaf(is_leaf), D(d), parent(nullptr) {
            entries.reserve(MAX_ENTRIES + 1);
        }

        Rect compute_mbr() const {
            Rect out(D);
            if (entries.empty()) return out;
            out = entries[0].mbr;
            for (size_t i = 1; i < entries.size(); ++i) out = rect_union(out, entries[i].mbr);
            return out;
        }
    };

    struct Neighbor {
        double dist_sq;
        Point p;
        bool operator<(const Neighbor& other) const { return dist_sq < other.dist_sq; } // max-heap
    };

    struct PQItem {
        double mindist_sq;
        Node* node;
        bool operator>(const PQItem& other) const { return mindist_sq > other.mindist_sq; }
    };

    Node* root;
    int D;
    long long node_count;

    // ---------- Rect helpers ----------
    static Rect rect_from_point(const Point& p) {
        Rect r(p.D);
        for (int i = 0; i < p.D; ++i) r.mn[i] = r.mx[i] = p.coords[i];
        return r;
    }

    static Rect rect_union(const Rect& a, const Rect& b) {
        Rect r(a.D);
        for (int i = 0; i < a.D; ++i) {
            r.mn[i] = std::min(a.mn[i], b.mn[i]);
            r.mx[i] = std::max(a.mx[i], b.mx[i]);
        }
        return r;
    }

    static double rect_volume(const Rect& r) {
        // hipervolumen (para puntos degenerados usamos EPS)
        double v = 1.0;
        for (int i = 0; i < r.D; ++i) {
            double len = std::max(EPS, r.mx[i] - r.mn[i]);
            v *= len;
        }
        return v;
    }

    static double rect_enlargement(const Rect& cur, const Rect& add) {
        Rect u = rect_union(cur, add);
        return rect_volume(u) - rect_volume(cur);
    }

    static double rect_minDist_sq(const Rect& r, const Point& q) {
        double dist = 0.0;
        for (int i = 0; i < r.D; ++i) {
            double qi = q.get(i);
            if (qi < r.mn[i]) {
                double d = r.mn[i] - qi; dist += d * d;
            } else if (qi > r.mx[i]) {
                double d = qi - r.mx[i]; dist += d * d;
            }
        }
        return dist;
    }

    // ---------- Memory management ----------
    Node* new_node(bool leaf) {
        node_count++;
        return new Node(leaf, D);
    }

    void delete_subtree(Node* n) {
        if (!n) return;
        if (!n->leaf) for (auto& e : n->entries) delete_subtree(e.child);
        delete n;
    }

    // ---------- Choose leaf ----------
    int choose_subtree_index(Node* n, const Rect& r) {
        int best = -1;
        double best_enl = std::numeric_limits<double>::max();
        double best_vol = std::numeric_limits<double>::max();

        for (int i = 0; i < (int)n->entries.size(); ++i) {
            const auto& e = n->entries[i];
            double enl = rect_enlargement(e.mbr, r);
            double vol = rect_volume(e.mbr);

            if (enl < best_enl || (enl == best_enl && vol < best_vol)) {
                best_enl = enl;
                best_vol = vol;
                best = i;
            }
        }
        return best;
    }

    Node* choose_leaf(Node* n, const Rect& r) {
        if (n->leaf) return n;
        int idx = choose_subtree_index(n, r);
        return choose_leaf(n->entries[idx].child, r);
    }

    // ---------- Split (Quadratic split) ----------
    std::pair<Node*, Node*> split_node(Node* n) {
        std::vector<Entry> E = std::move(n->entries);
        n->entries.clear();

        Node* g1 = n;
        Node* g2 = new_node(n->leaf);
        g2->parent = g1->parent;

        auto pick_seeds = [&]() -> std::pair<int,int> {
            int s1 = -1, s2 = -1;
            double worst = -1.0;
            for (int i = 0; i < (int)E.size(); ++i) {
                for (int j = i + 1; j < (int)E.size(); ++j) {
                    Rect u = rect_union(E[i].mbr, E[j].mbr);
                    double d = rect_volume(u) - rect_volume(E[i].mbr) - rect_volume(E[j].mbr);
                    if (d > worst) { worst = d; s1 = i; s2 = j; }
                }
            }
            return {s1, s2};
        };

        auto [i1, i2] = pick_seeds();
        if (i1 > i2) std::swap(i1, i2);

        Entry seed1 = E[i1];
        Entry seed2 = E[i2];
        E.erase(E.begin() + i2);
        E.erase(E.begin() + i1);

        g1->entries.push_back(seed1);
        g2->entries.push_back(seed2);

        while (!E.empty()) {
            // Forzar mínimos
            if ((int)g1->entries.size() + (int)E.size() == MIN_ENTRIES) {
                for (auto& e : E) g1->entries.push_back(e);
                E.clear();
                break;
            }
            if ((int)g2->entries.size() + (int)E.size() == MIN_ENTRIES) {
                for (auto& e : E) g2->entries.push_back(e);
                E.clear();
                break;
            }

            Rect mbr1 = g1->compute_mbr();
            Rect mbr2 = g2->compute_mbr();

            int best_idx = -1;
            double best_diff = -1.0;

            for (int i = 0; i < (int)E.size(); ++i) {
                double e1 = rect_enlargement(mbr1, E[i].mbr);
                double e2 = rect_enlargement(mbr2, E[i].mbr);
                double diff = std::fabs(e1 - e2);

                if (diff > best_diff) {
                    best_diff = diff;
                    best_idx = i;
                }
            }

            Entry pick = E[best_idx];
            E.erase(E.begin() + best_idx);

            Rect cur1 = g1->compute_mbr();
            Rect cur2 = g2->compute_mbr();
            double enl1 = rect_enlargement(cur1, pick.mbr);
            double enl2 = rect_enlargement(cur2, pick.mbr);

            if (enl1 < enl2) g1->entries.push_back(pick);
            else if (enl2 < enl1) g2->entries.push_back(pick);
            else {
                double vol1 = rect_volume(cur1);
                double vol2 = rect_volume(cur2);
                if (vol1 < vol2) g1->entries.push_back(pick);
                else if (vol2 < vol1) g2->entries.push_back(pick);
                else {
                    if (g1->entries.size() <= g2->entries.size()) g1->entries.push_back(pick);
                    else g2->entries.push_back(pick);
                }
            }
        }

        // Reparar parent pointers si es interno
        if (!g1->leaf) {
            for (auto& e : g1->entries) if (e.child) e.child->parent = g1;
            for (auto& e : g2->entries) if (e.child) e.child->parent = g2;
        }

        return {g1, g2};
    }

    void adjust_tree_after_insert(Node* leaf, Node* split_node_new) {
        Node* n = leaf;
        Node* nn = split_node_new;

        while (true) {
            if (n == root) {
                if (nn) {
                    Node* newRoot = new_node(false);
                    newRoot->leaf = false;

                    Rect m1 = n->compute_mbr();
                    Rect m2 = nn->compute_mbr();

                    newRoot->entries.emplace_back(m1, n, D);
                    newRoot->entries.emplace_back(m2, nn, D);

                    n->parent = newRoot;
                    nn->parent = newRoot;
                    root = newRoot;
                }
                break;
            }

            Node* parent = n->parent;

            // actualizar MBR de n en parent
            for (auto& e : parent->entries) {
                if (e.child == n) { e.mbr = n->compute_mbr(); break; }
            }

            if (nn) {
                Rect nn_mbr = nn->compute_mbr();
                parent->entries.emplace_back(nn_mbr, nn, D);
                nn->parent = parent;

                if ((int)parent->entries.size() > MAX_ENTRIES) {
                    auto [p1, p2] = split_node(parent);
                    n = p1;
                    nn = p2;
                    continue;
                }
            }

            n = parent;
            nn = nullptr;
        }
    }

    void insert_one(const Point& p) {
        if (!root) root = new_node(true);

        Rect r = rect_from_point(p);
        Node* leaf = choose_leaf(root, r);

        leaf->entries.emplace_back(r, p);

        Node* new_split = nullptr;
        if ((int)leaf->entries.size() > MAX_ENTRIES) {
            auto [g1, g2] = split_node(leaf);
            new_split = g2;
        }
        adjust_tree_after_insert(leaf, new_split);
    }

public:
    explicit RTree(int dimensions = INITIAL_DIM) : root(nullptr), D(dimensions), node_count(0) {}

    ~RTree() { delete_subtree(root); }

    void insert_batch(const std::vector<Point>& points) {
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& p : points) insert_one(p);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        std::cout << "Tiempo de Inserción (Batch, R-Tree): "
                  << std::fixed << std::setprecision(6)
                  << duration.count() << " segundos\n";
    }

    std::vector<Point> kNearestNeighbors(const Point& query, int k) const {
        if (!root || k <= 0) return {};

        std::priority_queue<Neighbor> best; // max-heap
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;

        auto start = std::chrono::high_resolution_clock::now();
        pq.push({0.0, root});

        while (!pq.empty()) {
            auto cur = pq.top();
            pq.pop();

            double worst = (best.size() < (size_t)k) ? std::numeric_limits<double>::max() : best.top().dist_sq;
            if (cur.mindist_sq > worst) break;

            Node* n = cur.node;
            if (n->leaf) {
                for (const auto& e : n->entries) {
                    double d = e.point.distanceSq(query);
                    if ((int)best.size() < k) best.push({d, e.point});
                    else if (d < best.top().dist_sq) {
                        best.pop();
                        best.push({d, e.point});
                    }
                }
            } else {
                for (const auto& e : n->entries) {
                    double md = rect_minDist_sq(e.mbr, query);
                    double worst2 = (best.size() < (size_t)k) ? std::numeric_limits<double>::max() : best.top().dist_sq;
                    if (md <= worst2) pq.push({md, e.child});
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Tiempo de Búsqueda (KNN, k=" << k << ", R-Tree): "
                  << std::fixed << std::setprecision(6)
                  << duration.count() << " segundos\n";

        std::vector<Point> res;
        while (!best.empty()) {
            res.push_back(best.top().p);
            best.pop();
        }
        std::reverse(res.begin(), res.end());
        return res;
    }

    long long getNodeCount() const { return node_count; }
    int getDimensions() const { return D; }
};
