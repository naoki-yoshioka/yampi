#ifndef YAMPI_ASSERTION_MODE_HPP
# define YAMPI_ASSERTION_MODE_HPP

# include <mpi.h>


namespace yampi
{
  enum class assertion_mode
    : int
  {
    no_check = MPI_MODE_NOCHECK, no_store = MPI_MODE_NOSTORE, no_put = MPI_MODE_NOPUT,
    no_precede = MPI_MODE_NOPRECEDE, no_succeed = MPI_MODE_NOSUCCEED
  };
}


#endif

