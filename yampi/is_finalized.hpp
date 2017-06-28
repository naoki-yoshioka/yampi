#ifndef YAMPI_IS_FINALIZED_HPP
# define YAMPI_IS_FINALIZED_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  inline bool is_finalized()
  {
    int result;
    int const error_code = MPI_Finalized(&result);

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_finalized");

    return result;
  }
}


#endif

