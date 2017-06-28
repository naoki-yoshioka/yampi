#ifndef YAMPI_IS_REQUEST_COMPLETE_HPP
# define YAMPI_IS_REQUEST_COMPLETE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/request.hpp>
# include <yampi/status.hpp>
# include <yampi/eror.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename Value>
  inline std::pair<bool, ::yampi::status> is_request_complete(::yampi::request const& request)
  {
    int flag;
    MPI_Status stat;
    int const error_code
      = MPI_Test(
          const_cast<MPI_Request*>(YAMPI_addressof(request.mpi_request())),
          &flag, YAMPI_addressof(stat));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_request_complete");
    return std::make_pair(static_cast<bool>(flag), ::yampi::status(stat));
  }

  inline bool is_request_complete(::yampi::request const& request, ::yampi::ignore_status_t const)
  {
    int flag;
    int const error_code
      = MPI_Test(
          const_cast<MPI_Request*>(YAMPI_addressof(request.mpi_request())),
          &flag, MPI_STATUS_IGNORE);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::is_request_complete");
    return static_cast<bool>(flag);
  }
}


# undef YAMPI_addressof

#endif

