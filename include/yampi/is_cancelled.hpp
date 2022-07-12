#ifndef YAMPI_IS_CANCELLED_HPP
# define YAMPI_IS_CANCELLED_HPP

# include <memory>

# include <mpi.h>

# include <yampi/status.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  inline bool is_cancelled(::yampi::status const& status, ::yampi::environment const& environment)
  {
    int flag;
    int const error_code
      = MPI_Test_cancelled(std::addressof(status.mpi_status()), std::addressof(flag));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error(error_code, "yampi::is_cancelled", environment);
  }
}


#endif

