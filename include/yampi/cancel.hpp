#ifndef YAMPI_CANCEL_HPP
# define YAMPI_CANCEL_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/request.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  inline void cancel(::yampi::request const& request, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Cancel(const_cast<MPI_Request*>(YAMPI_addressof(request.mpi_request())));

    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::cancel", environment);
  }
}


# undef YAMPI_addressof

#endif

