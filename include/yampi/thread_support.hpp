#ifndef YAMPI_THREAD_SUPPORT_HPP
# define YAMPI_THREAD_SUPPORT_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
  enum class thread_support : int
  {
    single = MPI_THREAD_SINGLE,
    funneled = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple = MPI_THREAD_MULTIPLE
  };

#   define YAMPI_THREAD_SUPPORT_TYPE yampi::thread_support
# else
  namespace thread_support
  {
    enum type
    {
      single = MPI_THREAD_SINGLE,
      funneled = MPI_THREAD_FUNNELED,
      serialized = MPI_THREAD_SERIALIZED,
      multiple = MPI_THREAD_MULTIPLE
    };
  }

#   define YAMPI_THREAD_SUPPORT_TYPE yampi::thread_support::type
# endif
}


#endif

