#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  template <typename Value>
  inline void broadcast(
    ::yampi::buffer<Value> buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Bcast(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }

  template <typename SendValue>
  inline void broadcast(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code
      = MPI_Bcast(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }

  inline void broadcast(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    auto const error_code = MPI_Bcast(nullptr, 0, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }
}


#endif

