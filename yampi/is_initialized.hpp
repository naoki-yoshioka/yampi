#ifndef YAMPI_IS_INITIALIZED_HPP
# define YAMPI_IS_INITIALIZED_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  inline bool is_initialized()
  {
    int result;
    int const error_code = MPI_Initialized(&result);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_initialized");

    return result;
  }
}


#endif

