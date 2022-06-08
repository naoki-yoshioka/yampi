#ifndef YAMPI_ASSERTION_MODE_HPP
# define YAMPI_ASSERTION_MODE_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class assertion_mode
    : int
  {
    no_check = MPI_MODE_NOCHECK, no_store = MPI_MODE_NOSTORE, no_put = MPI_MODE_NOPUT,
    no_precede = MPI_MODE_NOPRECEDE, no_succeed = MPI_MODE_NOSUCCEED
  };
# else // BOOST_NO_CXX11_SCOPED_ENUMS
  namespace assertion_mode
  {
    enum assertion_mode_
    {
      no_check = MPI_MODE_NOCHECK, no_store = MPI_MODE_NOSTORE, no_put = MPI_MODE_NOPUT,
      no_precede = MPI_MODE_NOPRECEDE, no_succeed = MPI_MODE_NOSUCCEED
    };
  }
# endif // BOOST_NO_CXX11_SCOPED_ENUMS
}


#endif

