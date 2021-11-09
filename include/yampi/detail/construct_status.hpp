#ifndef YAMPI_DETAIL_CONSTRUCT_STATUS_HPP
# define YAMPI_DETAIL_CONSTRUCT_STATUS_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/status.hpp>


namespace yampi
{
  namespace detail
  {
    struct construct_status
    {
      ::yampi::status operator()(MPI_Status const& mpi_status) const
      { return ::yampi::status(mpi_status); }
    };
  }
}


#endif

