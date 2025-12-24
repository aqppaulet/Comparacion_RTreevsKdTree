#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <queue>
#include <array>
#include <iomanip>
#include <string>
#include <fstream>

#ifdef __linux__
#include <malloc.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

// ===================== CONFIG =====================
constexpr int MAX_DIM = 8;
constexpr int INITIAL_DIM = 2;

// OJO: Mantengo EXACTO lo que tu KDTree usa.
const std::array<int, MAX_DIM> COLUMN_INDICES = {1, 3, 5, 16, 17, 2, 18, 19}; // columnas a extraer

// ===================== POINT ======================
struct Point {
    std::array<double, MAX_DIM> coords;
    int id; // identifica el punto original (fila)
    int D;  // dimensiones usadas

    Point(int dimensions = INITIAL_DIM) : id(-1), D(dimensions) {
        std::fill(coords.begin(), coords.end(), 0.0);
    }

    double get(int dim) const {
        if (dim < D) return coords[dim];
        return 0.0;
    }

    double distanceSq(const Point& other) const {
        double distSq = 0.0;
        for (int i = 0; i < D; ++i) {
            double diff = coords[i] - other.coords[i];
            distSq += diff * diff;
        }
        return distSq;
    }
};

// ===================== MEMORY ======================
long get_memory_usage_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        SIZE_T ram = pmc.WorkingSetSize;
        return static_cast<long>(ram / 1024);
    }
    return 0;
#else
    long vmrss = 0;
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::stringstream ss(line.substr(6));
            ss >> vmrss;
            break;
        }
    }
    return vmrss;
#endif
}

// ===================== CSV (rapidcsv) ======================
#include "rapidcsv.h"

std::vector<Point> read_csv(const std::string& filename, int dimensions) {
    // 0 → primera fila son headers, -1 → no hay labels de filas
    rapidcsv::Document doc(filename, rapidcsv::LabelParams(0, -1));

    std::vector<Point> points;
    int row_count = static_cast<int>(doc.GetRowCount());

    for (int r = 0; r < row_count; r++) {
        Point p(dimensions);
        p.id = r;

        bool valid = true;

        for (int i = 0; i < dimensions; i++) {
            int col = COLUMN_INDICES[i];

            try {
                // Mantengo tu mismo patrón (leer string y double)
                std::string raw = doc.GetCell<std::string>(col, r);
                (void)raw;

                double val = doc.GetCell<double>(col, r);
                p.coords[i] = val;
            } catch (const std::exception&) {
                valid = false;
                break;
            }
        }

        if (valid) {
            points.push_back(p);
        } else {
            std::cout << "  ✘ Punto descartado por error\n";
        }
    }

    std::cout << "\n=== Total puntos válidos: " << points.size() << " ===\n";
    return points;
}

// ===================================================
// Incluimos implementaciones (NO compilar aparte)
#include "kdtree.cpp"
#include "rtree.cpp"
// ===================================================

static void print_query(const Point& q, int D) {
    std::cout << "Coordenadas de Consulta: (";
    for (int i = 0; i < D; ++i) {
        std::cout << q.get(i) << (i == D - 1 ? "" : ", ");
    }
    std::cout << ")\n";
}

static void print_knn_results(const std::string& title,
                              const std::vector<Point>& neighbors,
                              const Point& query_point,
                              int D)
{
    std::cout << "\nResultados de KNN (" << title << "):\n";
    for (size_t i = 0; i < neighbors.size(); ++i) {
        double dist_sq = neighbors[i].distanceSq(query_point);
        std::cout << "  " << i + 1 << ". ID: " << neighbors[i].id
                  << ", Distancia Cuadrada: " << std::fixed << std::setprecision(4) << dist_sq
                  << ", Coordenadas: (";
        for (int j = 0; j < D; ++j) {
            std::cout << neighbors[i].get(j) << (j == D - 1 ? "" : ", ");
        }
        std::cout << ")\n";
    }
}

void run_comparison_test(int dimensions, const std::string& filename) {
    std::cout << "\n==================================================\n";
    std::cout << "  Comparación KD-Tree vs R-Tree con D = " << dimensions << " dimensiones\n";
    std::cout << "==================================================\n";

    // 1) Memoria inicial
    long mem_before = get_memory_usage_kb();
    std::cout << "Uso de Memoria (Antes de la Carga): " << mem_before << " KB\n";

    // 2) Carga de datos
    std::vector<Point> points = read_csv(filename, dimensions);
    if (points.empty()) return;

    long mem_after_load = get_memory_usage_kb();
    std::cout << "Uso de Memoria (Después de la Carga de Datos): " << mem_after_load << " KB\n";
    std::cout << "Memoria usada por los datos: " << (mem_after_load - mem_before) << " KB\n";

    // Punto de consulta = primer punto del dataset (MISMO para ambos)
    int k = 200;
    Point query_point = points[0];

    std::cout << "\nBúsqueda de " << k << " Vecinos Más Cercanos... Punto de Consulta (ID "
              << query_point.id << "):\n";
    print_query(query_point, dimensions);

    // ================= KD-TREE =================
    long base = get_memory_usage_kb();
    std::cout << "\n-------------------- KD-TREE --------------------\n";
    std::vector<Point> points_for_kd = points; // KD ordena internamente; copiamos para no alterar dataset base
    KDTree kd(dimensions);
    kd.insert_batch(points_for_kd);

    long mem_after_kd = get_memory_usage_kb();
    std::cout << "Uso de Memoria (Después de la Construcción del Árbol): " << mem_after_kd << " KB\n";
    std::cout << "Memoria usada por el Árbol KD: " << (mem_after_kd - base) << " KB\n";
    std::cout << "Número de Nodos Creados: " << kd.getNodeCount() << "\n";

    std::vector<Point> knn_kd = kd.kNearestNeighbors(query_point, k);
    print_knn_results("KD-Tree", knn_kd, query_point, dimensions);
    
    // (opcional) liberar heap al SO (Linux)
    malloc_trim(0);
    #endif

    // ================= R-TREE =================
    base = get_memory_usage_kb();

    std::cout << "\n-------------------- R-TREE ---------------------\n";
    RTree rt(dimensions);
    rt.insert_batch(points);

    long mem_after_rt = get_memory_usage_kb();
    std::cout << "Uso de Memoria (Después de la Construcción del Árbol): " << mem_after_rt << " KB\n";
    std::cout << "Memoria usada por el Árbol R: " << (mem_after_rt - base) << " KB\n";
    std::cout << "Número de Nodos Creados: " << rt.getNodeCount() << "\n";

    std::vector<Point> knn_rt = rt.kNearestNeighbors(query_point, k);
    print_knn_results("R-Tree", knn_rt, query_point, dimensions);
}   

// --- MAIN (mismo menú) ---
int main() {
    std::string filename = "prod.csv";

    while (true) {
        std::cout << "\n=============================\n";
        std::cout << "      MENU DE OPCIONES\n";
        std::cout << "=============================\n";
        std::cout << "1. Probar KD-Tree con 2 dimensiones\n";
        std::cout << "2. Probar KD-Tree con 6 dimensiones\n";
        std::cout << "3. Probar KD-Tree con N dimensiones\n";
        std::cout << "4. Ver memoria RAM usada actualmente\n";
        std::cout << "0. Salir\n";
        std::cout << "Seleccione una opción: ";

        int opcion;
        std::cin >> opcion;

        if (opcion == 0) {
            std::cout << "Saliendo...\n";
            break;
        }

        int dims;
        switch (opcion) {
            case 1:
                run_comparison_test(2, filename);
                break;
            case 2:
                run_comparison_test(6, filename);
                break;
            case 3:
                std::cout << "Ingrese cantidad de dimensiones (1-" << MAX_DIM << "): ";
                std::cin >> dims;
                if (dims < 1 || dims > MAX_DIM) {
                    std::cout << "ERROR: Dimensiones fuera de rango.\n";
                } else {
                    run_comparison_test(dims, filename);
                }
                break;
            case 4: {
                long mem = get_memory_usage_kb();
                std::cout << "Memoria actual usada por el proceso: " << mem << " KB\n";
                break;
            }
            default:
                std::cout << "Opción no válida.\n";
        }
    }
    return 0;
}
