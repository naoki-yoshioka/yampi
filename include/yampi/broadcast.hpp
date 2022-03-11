#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

# include <boost/config.hpp>

# include <mpi.h>

# ifdef BOOST_NO_CXX11_NULLPTR
#   include <cstddef>
# endif

# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif


namespace yampi
{
  template <typename Value>
  inline void broadcast(
    ::yampi::buffer<Value> buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    int const error_code
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
    int const error_code
      = MPI_Bcast(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }

  inline void broadcast(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Bcast(nullptr, 0, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
  }
}


# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif

#endif

