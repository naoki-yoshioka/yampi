#ifndef YAMPI_COMPLETE_EXCHANGE_HPP
# define YAMPI_COMPLETE_EXCHANGE_HPP

# include <cassert>
# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# if MPI_VERSION >= 3
#   include <yampi/topology.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/immediate_request.hpp>
# if MPI_VERSION >= 4
#   include <yampi/persistent_request.hpp>
#   include <yampi/information.hpp>
# endif // MPI_VERSION >= 4


namespace yampi
{
  // Blocking complete-exchange
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.size(environment) <= send_buffer.data());
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }

  template <typename Value>
  inline void complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoall_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoall(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }

  // only for intercommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.remote_size(environment) <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.remote_size(environment) <= send_buffer.data());
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

    auto const remote_size = communicator.remote_size(environment);
    auto const receive_count = receive_buffer.count() / remote_size;
    assert(receive_count * remote_size == receive_buffer.count());

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }
# if MPI_VERSION >= 3

  // neighbor complete_exchange
  template <typename SendValue, typename ContiguousIterator, typename Topology>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * topology.num_neighbors(environment) <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * topology.num_neighbors(environment) <= send_buffer.data());
# endif // MPI_VERSION >= 4

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

    auto const num_neighbors = topology.num_neighbors(environment);
    auto const receive_count = receive_buffer.count() / num_neighbors;
    assert(receive_count * num_neighbors == receive_buffer.count());

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
  }
# endif // MPI_VERSION >= 3

  // Nonblocking complete-exchange
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.size(environment) <= send_buffer.data());
# endif // MPI_VERSION >= 4

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request, ::yampi::buffer<Value> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.remote_size(environment) <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * communicator.remote_size(environment) <= send_buffer.data());
# endif // MPI_VERSION >= 4

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
# else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
# endif // MPI_VERSION >= 4

    auto const remote_size = communicator.remote_size(environment);
    auto const receive_count = receive_buffer.count() / remote_size;
    assert(receive_count * remote_size == receive_buffer.count());

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 3

  // neighbor complete_exchange
  template <typename SendValue, typename ContiguousIterator, typename Topology>
  inline void complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
#   if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * topology.num_neighbors(environment) <= send_buffer.data());
#   else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count() * topology.num_neighbors(environment) <= send_buffer.data());
#   endif // MPI_VERSION >= 4

    MPI_Request mpi_request;
#   if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
#   if MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());
#   else // MPI_VERSION >= 4
    assert(send_buffer.data() + send_buffer.count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count() <= send_buffer.data());
#   endif // MPI_VERSION >= 4

    auto const num_neighbors = topology.num_neighbors(environment);
    auto const receive_count = receive_buffer.count() / num_neighbors;
    assert(receive_count * num_neighbors == receive_buffer.count());

    MPI_Request mpi_request;
#   if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoall_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoall(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count, receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
#   endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 3
# if MPI_VERSION >= 4

  // Persistent complete-exchange
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());

    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> receive_buffer, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.remote_size(environment) <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());

    auto const remote_size = communicator.remote_size(environment);
    auto const receive_count = receive_buffer.count() / remote_size;
    assert(receive_count * remote_size == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // neighbor complete_exchange
  template <typename SendValue, typename ContiguousIterator, typename Topology>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::information const& information,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * topology.num_neighbors(environment) <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::information const& information,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());

    auto const num_neighbors = topology.num_neighbors(environment);
    auto const receive_count = receive_buffer.count() / num_neighbors;
    assert(receive_count * num_neighbors == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  // only for intracommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.size(environment) <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());

    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request, ::yampi::buffer<Value> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    auto const size = communicator.size(environment);
    auto const receive_count = receive_buffer.count() / size;
    assert(receive_count * size == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue, typename ContiguousIterator>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * communicator.remote_size(environment) <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());

    auto const remote_size = communicator.remote_size(environment);
    auto const receive_count = receive_buffer.count() / remote_size;
    assert(receive_count * remote_size == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // neighbor complete_exchange
  template <typename SendValue, typename ContiguousIterator, typename Topology>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    static_assert(
      std::is_same<typename std::iterator_traits<ContiguousIterator>::value_type, SendValue>::value,
      "value_type of ContiguousIterator must be the same to SendValue");
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= std::addressof(*first) or std::addressof(*first) + send_buffer.count().mpi_count() * topology.num_neighbors(environment) <= send_buffer.data());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          std::addressof(*first), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() + send_buffer.count().mpi_count() <= receive_buffer.data() or receive_buffer.data() + receive_buffer.count().mpi_count() <= send_buffer.data());

    auto const num_neighbors = topology.num_neighbors(environment);
    auto const receive_count = receive_buffer.count() / num_neighbors;
    assert(receive_count * num_neighbors == receive_buffer.count());

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoall_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_count.mpi_count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::complete_exchange", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif
