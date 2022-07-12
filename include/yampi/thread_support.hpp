#ifndef YAMPI_THREAD_SUPPORT_HPP
# define YAMPI_THREAD_SUPPORT_HPP

# include <mpi.h>


namespace yampi
{
  enum class thread_support : int
  {
    single = MPI_THREAD_SINGLE,
    funneled = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple = MPI_THREAD_MULTIPLE
  };
}


#endif

