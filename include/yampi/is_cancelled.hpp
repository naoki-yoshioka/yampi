#ifndef YAMPI_IS_CANCELLED_HPP
# define YAMPI_IS_CANCELLED_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/status.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  inline bool is_cancelled(::yampi::status const& status, ::yampi::environment const& environment)
  {
    int flag;
    int const error_code
      = MPI_Test_cancelled(YAMPI_addressof(status.mpi_status()), YAMPI_addressof(flag));

    return error_code == MPI_SUCCESS
      ? static_cast<bool>(flag)
      : throw ::yampi::error(error_code, "yampi::is_cancelled", environment);
  }
}


# undef YAMPI_addressof

#endif

