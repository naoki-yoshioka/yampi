#ifndef YAMPI_BARRIER_HPP
# define YAMPI_BARRIER_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/communicator_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  inline void barrier(::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Barrier(communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::barrier", environment);
  }
}


#endif

