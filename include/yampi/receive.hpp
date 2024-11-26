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
# include <yampi/immediate_request.hpp>
# include <yampi/persistent_request.hpp>


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
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::receive", environment};
  }

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::receive", environment};
  }

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::receive", environment};
  }

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::receive", environment};
  }
# if MPI_VERSION >= 3

  template <typename Value>
  inline ::yampi::status receive(
    ::yampi::buffer<Value> buffer, ::yampi::message& message,
    ::yampi::environment const& environment)
  {
    MPI_Status stat;
#   if MPI_VERSION >= 4
    auto const error_code
      = MPI_Mrecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), std::addressof(stat));
#   else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Mrecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), std::addressof(stat));
#   endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::receive", environment};
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
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
  }

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
  }

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
  }

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::buffer<Value> buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
  }
# if MPI_VERSION >= 3

  template <typename Value>
  inline void receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::message& message,
    ::yampi::environment const& environment)
  {
#   if MPI_VERSION >= 4
    auto const error_code
      = MPI_Mrecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), MPI_STATUS_IGNORE);
#   else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Mrecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), MPI_STATUS_IGNORE);
#   endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
  }
# endif // MPI_VERSION >= 3

  // Nonblocking receive
  template <typename Value>
  inline void receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::immediate_request& request, ::yampi::buffer<Value> buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Irecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 3

  template <typename Value>
  inline void receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::message& message,
    ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
#   if MPI_VERSION >= 4
    auto const error_code
      = MPI_Imrecv_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Imrecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          std::addressof(message.mpi_message()), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }
#endif // MPI_VERSION >= 3

  // Persistent receive
  template <typename Value>
  inline void receive(
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void receive(
    ::yampi::persistent_request& request, ::yampi::buffer<Value> buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Recv_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG, communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::receive", environment};
    request.reset(mpi_request, environment);
  }
}


#endif

