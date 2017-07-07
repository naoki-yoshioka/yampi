#ifndef YAMPI_LOWEST_IO_PROCESS_HPP
# define YAMPI_LOWEST_IO_PROCESS_HPP

# include <boost/config.hpp>

# ifdef __FUJITSU // needed for combination of Boost 1.61.0 and Fujitsu compiler
#   include <boost/utility/in_place_factory.hpp>
#   include <boost/utility/typed_in_place_factory.hpp>
# endif
# include <boost/optional.hpp>
# include <boost/none.hpp>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/rank.hpp>
# include <yampi/communicator.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  // return the rank that is lowest in the group of world communicator
  inline boost::optional< ::yampi::rank > lowest_io_process(::yampi::environment const& environment)
  {
    BOOST_CONSTEXPR_OR_CONST ::yampi::rank zero_rank = ::yampi::rank(0);
    ::yampi::rank const io_process = ::yampi::io_process(environment);
    if (io_process == ::yampi::any_source())
      return zero_rank;
    else if (io_process == ::yampi::null_process())
      return boost::none;

    // TODO: implement yampi::all_reduce and replace the following statements to it
    int result;
    int const error_code
      = MPI_Allreduce(
          const_cast<int*>(&io_process.mpi_rank()), &result, 1, MPI_INT, MPI_MIN,
          ::yampi::world_communicator().mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::lowest_io_process", environment);

    return ::yampi::rank(result);
  }
}


#endif
