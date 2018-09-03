#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename Value>
  inline void broadcast(
    ::yampi::communicator const& communicator, ::yampi::rank const root,
    ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer)
  {
    int const error_code
      = MPI_Bcast(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }

  template <typename Value>
  inline void broadcast(
    ::yampi::communicator const& communicator, ::yampi::rank const root,
    ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer)
  {
    int const error_code
      = MPI_Bcast(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }
}


#endif

