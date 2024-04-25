#include <iostream>
#include <vector>

// Function to calculate the Halton sequence for a specific index and base
double halton(int index, int base) {
    double result = 0;
    double f = 1.0 / base;
    int i = index;
    while (i > 0) {
        result += f * (i % base);
        i /= base;
        f /= base;
    }
    return result;
}

int main() {
    int numPoints = 10;  // Number of points in the Halton sequence

    std::cout << "First " << numPoints << " points of the Halton sequence in 2D:" << std::endl;
    for (int i = 1; i <= numPoints; ++i) {
        double x = halton(i, 2);  // Base 2 for the first dimension
        double y = halton(i, 3);  // Base 3 for the second dimension
        std::cout << "(" << x << ", " << y << ")" << std::endl;
    }

    return 0;
}