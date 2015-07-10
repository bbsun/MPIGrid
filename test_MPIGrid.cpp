
#include "gtest/gtest.h"
#include <mpi.h>
#include "MPIGrid.hpp"
#include "assert.h"
#include <iomanip>

/*
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
*/

TEST(MPIGridTest, Scatter_2D)
{
    int ndims = 2;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 8;
    global_dims[1] = 20;
    np_dims[0] = 2;
    np_dims[1] = 2;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, local_dims, np_dims, ndims);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]*local_dims[1]);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]*local_dims[1]; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    ASSERT_EQ(err, 0);

    err = grid.gather(gathered_data, local_data);
    ASSERT_EQ(err, 0);

    if (rank == 0) {
        for (int i=0; i<global_dims[0]*global_dims[1]; i++)
            EXPECT_EQ(global_data[i], gathered_data[i]);
    }

    free(global_data);
    free(gathered_data);
    free(local_data);

}

int main(int argc, char ** argv)
{
    /*
    MPI_Init(&argc, &argv);
    int ndims = 2;
    int global_dims[ndims];
    int local_dims[ndims];
    int np_dims[ndims];

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 8;
    global_dims[1] = 20;
    np_dims[0] = 1;
    np_dims[1] = 4;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, local_dims, np_dims, ndims);
    //ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*local_dims[0]*local_dims[1]);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<local_dims[0]*local_dims[1]; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    //ASSERT_EQ(err, 0);

    MPI_Status status;

    int file_free = 0;
    if (rank == 0) file_free = 1;
    else MPI_Recv(&file_free, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);

    for (int i=0; i<local_dims[0]; i++)
    {
    for (int j=0; j<local_dims[1]; j++)
    {
        int ind = i*local_dims[1] + j;
        std::cout << std::setw(4) << local_data[ind] << " ";
    }
        std::cout << std::endl;
    }


    if (rank != np-1) MPI_Send(&file_free, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);

    err = grid.gather(gathered_data, local_data);

    if (rank == 0) {
    for (int i=0; i<global_dims[0]; i++)
    {
    for (int j=0; j<global_dims[1]; j++)
    {
        int ind = i*global_dims[1] + j;
        std::cout << std::setw(4) << gathered_data[ind] << " ";
    }
    std::cout << std::endl;
    }
    }



    free(global_data);
    free(gathered_data);
    free(local_data);
    MPI_Finalize();
    return 0;
    */

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;


}
