#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/immediate_request.hpp>
# if MPI_VERSION >= 4
#   include <yampi/persistent_request.hpp>
#   include <yampi/information.hpp>
# endif // MPI_VERSION >= 4


namespace yampi
{
  // Blocking broadcast
  template <typename Value>
  inline void broadcast(
    ::yampi::buffer<Value> buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Bcast_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Bcast(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
  }

  template <typename SendValue>
  inline void broadcast(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Bcast_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Bcast(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
  }

  inline void broadcast(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code = MPI_Bcast_c(nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code = MPI_Bcast(nullptr, 0, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
  }

  // Nonblocking broadcast
  template <typename Value>
  inline void broadcast(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ibcast_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ibcast(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::broadcast", environment);
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void broadcast(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ibcast_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ibcast(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }

  inline void broadcast(
    ::yampi::immediate_request& request, ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ibcast_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ibcast(
          nullptr, 0, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 4

  // Persistent broadcast
  template <typename Value>
  inline void broadcast(
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Bcast_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void broadcast(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Bcast_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }

  inline void broadcast(
    ::yampi::persistent_request& request, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Bcast_init_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm(), information.mpi_info(),
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  template <typename Value>
  inline void broadcast(
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Bcast_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void broadcast(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Bcast_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }

  inline void broadcast(
    ::yampi::persistent_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Bcast_init_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, MPI_PROC_NULL, communicator.mpi_comm(), MPI_INFO_NULL,
          std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::broadcast", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif

