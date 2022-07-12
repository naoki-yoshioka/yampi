#ifndef YAMPI_ADDRESSOF_HPP
# define YAMPI_ADDRESSOF_HPP

# include <memory>

# include <mpi.h>

# include <yampi/address.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename Value>
  inline ::yampi::address addressof(Value& value, ::yampi::environment const& environment)
  {
    MPI_Aint mpi_address;
    int const error_code = MPI_Get_address(std::addressof(value), std::addressof(mpi_address));
    return error_code == MPI_SUCCESS
      ? ::yampi::address(mpi_address)
      : throw ::yampi::error(error_code, "yampi::address", environment);
  }
}


#endif
