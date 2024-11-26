#ifndef YAMPI_SEND_RECEIVE_HPP
# define YAMPI_SEND_RECEIVE_HPP

# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/cartesian.hpp>
# include <yampi/immediate_request.hpp>


namespace yampi
{
  // Blocking send-receive
  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  // Blocking send-receive, with replacement
  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Status stat;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(stat));
# endif // MPI_VERSION >= 4
    return error_code == MPI_SUCCESS
      ? ::yampi::status{stat}
      : throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  // Blocking send-receive, ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Sendrecv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION
    auto const error_code
      = MPI_Sendrecv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  // Blocking send-receive, with replacement, ignoring status
  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Sendrecv_replace(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), MPI_STATUS_IGNORE);
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
  }
# if MPI_VERSION >= 4

  // Nonblocking send-receive
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  // Nonblocking send-receive, with replacement
  template <typename Value>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source, ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::rank const source,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          source.mpi_rank(), MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::tag const receive_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, receive_tag.mpi_tag(),
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Isendrecv_replace_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), send_tag.mpi_tag(),
          MPI_ANY_SOURCE, MPI_ANY_TAG,
          communicator.mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4

  /* Cartesian versions */
  // Blocking send-receive
  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    return ::yampi::send_receive(
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline ::yampi::status send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    return ::yampi::send_receive(
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source),
      cartesian.communicator(), environment);
  }

  // Blocking send-receive, with replacement
  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<Value> buffer, ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    return ::yampi::send_receive(
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline ::yampi::status send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<Value> buffer, ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    return ::yampi::send_receive(
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source),
      cartesian.communicator(), environment);
  }

  // Blocking send-receive, ignoring status
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      ignore_status,
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      ignore_status,
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source),
      cartesian.communicator(), environment);
  }

  // Blocking send-receive, with replacement, ignoring status
  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<Value> buffer, ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      ignore_status,
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::ignore_status_t const ignore_status,
    ::yampi::cartesian_shift const& shift,
    ::yampi::buffer<Value> buffer, ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      ignore_status,
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source),
      cartesian.communicator(), environment);
  }
# if MPI_VERSION >= 4

  // Nonblocking send-receive
  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      request,
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::tag const send_tag,
    ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::cartesian const& cartesian,
    ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      request,
      send_buffer, ::yampi::rank(mpi_destination), send_tag,
      receive_buffer, ::yampi::rank(mpi_source),
      cartesian.communicator(), environment);
  }

  // Nonblocking send-receive, with replacement
  template <typename Value>
  inline void send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::tag const send_tag, ::yampi::tag const receive_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      request,
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source), receive_tag,
      cartesian.communicator(), environment);
  }

  template <typename Value>
  inline void send_receive(
    ::yampi::cartesian_shift const& shift,
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> buffer, ::yampi::tag const send_tag,
    ::yampi::cartesian const& cartesian, ::yampi::environment const& environment)
  {
    int mpi_source;
    int mpi_destination;
    auto const error_code
      = MPI_Cart_shift(
          cartesian.communicator().mpi_comm(), shift.direction(), shift.displacement(),
          std::addressof(mpi_source), std::addressof(mpi_destination));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send_receive", environment};

    ::yampi::send_receive(
      request,
      buffer, ::yampi::rank(mpi_destination), send_tag,
      ::yampi::rank(mpi_source),
      cartesian.communicator(), environment);
  }
# endif // MPI_VERSION >= 4
}


#endif

