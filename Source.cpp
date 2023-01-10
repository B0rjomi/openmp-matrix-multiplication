#include <iostream>
#include <omp.h>
#include <chrono>
using namespace std::chrono;

constexpr auto N = 200;


void fill_random(int aMatrix[N][N])
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            aMatrix[i][j] = rand();
        }
    }
}


void multiply(int aMatrix[N][N], int bMatrix[N][N], int product[N][N])
{
#pragma omp parallel for collapse(12)
    for (int row = 0; row < N; row++) 
    {
        for (int col = 0; col < N; col++)
        {
            // Multiply the row of A by the column of B to get the row, column of product.
            for (int inner = 0; inner < N; inner++)
            {
                product[row][col] += aMatrix[row][inner] * bMatrix[inner][col];
            }
        }
    }
}

int main()
{
    //define your matrices in main
    int aMatrix[N][N];
    int bMatrix[N][N];
    int product[N][N] = {};

    //fill them with random numbers
    fill_random(aMatrix);
    fill_random(bMatrix);

    //measure the multiply only
    const auto start = high_resolution_clock::now();
    omp_set_num_threads(16);

    multiply(aMatrix, bMatrix, product);

    const auto stop = high_resolution_clock::now();
    const auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Time taken by matrix multiplication: " << duration.count() << " milliseconds" << std::endl;

}