#ifndef YAMPI_ADDRESSOF_HPP
# define YAMPI_ADDRESSOF_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/address.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename Value>
  inline ::yampi::address addressof(Value& value, ::yampi::environment const& environment)
  {
    MPI_Aint mpi_address;
    int const error_code
      = MPI_Get_address(
          const_cast<Value*>(YAMPI_addressof(value)), YAMPI_addressof(mpi_address));
    return error_code == MPI_SUCCESS
      ? ::yampi::address(mpi_address)
      : throw ::yampi::error(error_code, "yampi::address", environment);
  }
}


# undef YAMPI_addressof

#endif
