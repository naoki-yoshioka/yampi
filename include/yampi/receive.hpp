#ifndef YAMPI_RECEIVE_HPP
# define YAMPI_RECEIVE_HPP

# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/message.hpp>


namespace yampi
{
  // Blocking receive
  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    int const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    int const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status(stat)
      : throw ::yampi::error(error_code, "yampi::receive", environment);
  }

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  { return ::yampi::receive(buffer, source, ::yampi::any_tag, communicator, environment); }

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  { return ::yampi::receive(buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
# if MPI_VERSION >= 3

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer, ::yampi::message& message,
    ::yampi::environment const& environment)
  {
    MPI_Status stat;
#   if MPI_VERSION >= 4
    int const error_code
      = MPI_Mrecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), std::addressof(stat));
#   else // MPI_VERSION >= 4
    int const error_code
      = MPI_Mrecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), std::addressof(stat));
#   endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status(stat)
      : throw ::yampi::error(error_code, "yampi::receive", environment);
  }
# endif // MPI_VERSION >= 3

  // Blocking receive (ignoring status)
  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    int const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    int const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::receive", environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  { ::yampi::receive(ignore_status, buffer, source, ::yampi::any_tag, communicator, environment); }

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value> buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  { ::yampi::receive(ignore_status, buffer, ::yampi::any_source, ::yampi::any_tag, communicator, environment); }
# if MPI_VERSION >= 3

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::message& message,
    ::yampi::environment const& environment)
  {
#   if MPI_VERSION >= 4
    int const error_code
      = MPI_Mrecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), MPI_STATUS_IGNORE);
#   else // MPI_VERSION >= 4
    int const error_code
      = MPI_Mrecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), MPI_STATUS_IGNORE);
#   endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::receive", environment);
  }
# endif // MPI_VERSION >= 3
}


#endif

