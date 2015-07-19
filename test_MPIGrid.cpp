
#include "gtest/gtest.h"
#include <mpi.h>
#include "MPIGrid.hpp"
#include "assert.h"
#include <iomanip>


void print_grid(double * data, int * dims)
{
    for (int i=0; i<dims[0]; i++)
    {
        if (i==1 || i== dims[0]-1 ) 
            for (int j=0; j<dims[1]; j++)
                std::cout << std::setw(4) << "----" << " ";
        std::cout << std::endl;

        for (int j=0; j<dims[1]; j++)
        {
            if (j==1 || j==dims[1]-1) std::cout << "|";
            else std::cout << " ";
            int ind = i*dims[1] + j;
            std::cout << std::setw(3) << data[ind] << " ";
        }
        std::cout << std::endl;
    }
}

TEST(MPIGridTest, Scatter_1D)
{
    int ndims = 1;
    int global_dims[ndims];
    int np_dims[ndims];
    int alloc_local;

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    global_dims[0] = 100;
    np_dims[0] = np;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 1, alloc_local);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]);
    double * local_data = (double *) malloc(sizeof(double)*alloc_local);

    for (int i=0; i<global_dims[0]; i++)
        global_data[i] = i+1;

    grid.scatter(global_data, local_data);

    for (int i=1; i<alloc_local-1; i++)
        EXPECT_EQ(local_data[i], rank*global_dims[0]/np + i);
}

TEST(MPIGridTest, Share_2D)
{
    int ndims = 2;
    int global_dims[ndims];
    int np_dims[ndims];
    int alloc_local;

    global_dims[0] = 4;
    global_dims[1] = 6;
    np_dims[0] = 2;
    np_dims[1] = 2;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 1, alloc_local);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*alloc_local);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<alloc_local; i++) local_data[i] = -1;

    err = grid.scatter(global_data, local_data);
    ASSERT_EQ(err, 0);
    err = grid.share(local_data);
    ASSERT_EQ(err, 0);

    if (rank == 0) {
        EXPECT_EQ(local_data[0], 24);
        EXPECT_EQ(local_data[1], 19);
        EXPECT_EQ(local_data[2], 20);
        EXPECT_EQ(local_data[3], 21);
        EXPECT_EQ(local_data[4], 22);
        EXPECT_EQ(local_data[5], 6);
        EXPECT_EQ(local_data[10], 12);
    } else if (rank == 3) {
        EXPECT_EQ(local_data[0], 9);
        EXPECT_EQ(local_data[1], 10);
        EXPECT_EQ(local_data[2], 11);
        EXPECT_EQ(local_data[3], 12);
        EXPECT_EQ(local_data[4], 7);
        EXPECT_EQ(local_data[5], 15);
        EXPECT_EQ(local_data[10], 21);
        EXPECT_EQ(local_data[15], 3);
        EXPECT_EQ(local_data[16], 4);
    }

    err = grid.gather(gathered_data, local_data);
    ASSERT_EQ(err, 0);

    if (rank==0) {
    for (int i=0; i<global_dims[0]*global_dims[1]; i++)
        EXPECT_EQ(gathered_data[i], global_data[i]);
    }

    free(global_data);
    free(local_data);
    free(gathered_data);
}

TEST(MPIGridTest, Scatter_2D)
{
    int ndims = 2;
    int global_dims[ndims];
    int np_dims[ndims];
    int alloc_local;

    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    global_dims[0] = 8;
    global_dims[1] = 20;
    np_dims[0] = 2;
    np_dims[1] = 2;

    MPIGrid grid;
    int err = grid.setup(MPI_COMM_WORLD, global_dims, np_dims, ndims, 1, alloc_local);
    ASSERT_EQ(err, 0);

    double * global_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);
    double * local_data = (double *) malloc(sizeof(double)*alloc_local);
    double * gathered_data = (double *) malloc(sizeof(double)*global_dims[0]*global_dims[1]);

    if (rank==0)
    for (int i=0; i<global_dims[0]*global_dims[1]; i++) global_data[i] = i+1;

    for (int i=0; i<alloc_local; i++) local_data[i] = -1;

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
    global_dims[0] = 16;
    global_dims[1] = 8;
    np_dims[0] = 2;
    np_dims[1] = 2;

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
    err = grid.share(local_data);

    MPI_Status status;

    int file_free = 0;
    if (rank == 0) file_free = 1;
    else MPI_Recv(&file_free, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &status);

    for (int i=0; i<local_dims[0]; i++)
    {
        if (i==MPIGRID_NROWS || i== local_dims[0]-MPIGRID_NROWS ) for (int j=0; j<local_dims[1]; j++)
        std::cout << std::setw(4) << "----" << " ";
        std::cout << std::endl;

        for (int j=0; j<local_dims[1]; j++)
        {
            if (j==MPIGRID_NROWS || j==local_dims[1]-MPIGRID_NROWS) std::cout << "|";
            else std::cout << " ";
            int ind = i*local_dims[1] + j;
            std::cout << std::setw(3) << local_data[ind] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "================================" << std::endl;


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
