
#include "gtest/gtest.h"
#include <mpi.h>
#include "MPIGrid.hpp"
#include "assert.h"

TEST(MPIGridTest, Scatter_1D)
{
    int ndims = 1;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    global_dims[0] = 100;
    np_dims[0] = np;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, local_dims, np_dims, ndims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]);

    for (int i=0; i<global_dims[0]; i++)
        global_data[i] = i+1;

    grid.scatter(global_data, local_data);

    for (int i=1; i<local_dims[0]-1; i++)
        EXPECT_EQ(local_data[i], rank*global_dims[0]/np + i);
}

int main(int argc, char ** argv)
{

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;


/*

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, local_dims, np_dims, 1);
    assert(err == 0);




    for (int i=0; i<local_dims[0]; i++)
        std::cout << local_data[i] << " ";

    std::cout << std::endl;

    */

}
