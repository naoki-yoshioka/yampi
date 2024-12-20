#ifndef YAMPI_NONCONTIGUOUS_COMPLETE_EXCHANGE_HPP
# define YAMPI_NONCONTIGUOUS_COMPLETE_EXCHANGE_HPP

# include <memory>

# include <mpi.h>

# include <yampi/noncontiguous_buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/communicator.hpp>
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
  // Blocking noncontiguous-complete-exchange
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoallv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoallv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoallv(
          const_cast<SendValue*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
  }

  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, true> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoallw_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoallw(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoallw(
          const_cast<SendValue*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.byte_displacement_first()), reinterpret_cast<MPI_Datatype*>(send_buffer.datatype_first()),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype*>(receive_buffer.datatype_first()),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoallv_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoallv(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
  }

  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, true> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Alltoallw_c(
          MPI_IN_PLACE, nullptr, nullptr, nullptr,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoallw(
          MPI_IN_PLACE, nullptr, nullptr, nullptr,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Alltoallw(
          MPI_IN_PLACE, nullptr, nullptr, nullptr,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype*>(receive_buffer.datatype_first()),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
  }
# if MPI_VERSION >= 3

  // neighbor noncontiguous_complete_exchange
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoallv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoallv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoallw_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          topology.communicator().mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_alltoallw(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.byte_displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          topology.communicator().mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
  }
# endif // MPI_VERSION >= 3

  // Nonblocking noncontiguous-complete-exchange
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, true> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallw_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallw(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallv_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallv(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<Value, true> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallw_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ialltoallw(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 3

  // neighbor noncontiguous_complete_exchange
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoallv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoallv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoallw_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_alltoallw(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 3
# if MPI_VERSION >= 4

  // Persistent noncontiguous-complete-exchange
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::information const& information,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, true> receive_buffer,
    ::yampi::information const& information,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallw_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallv_init_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, true> receive_buffer, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallw_init_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // neighbor noncontiguous_complete_exchange
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::information const& information,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoallv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::information const& information,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoallw_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          topology.communicator().mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, true> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallw_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallv_init_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_complete_exchange(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, true> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Alltoallw_init_c(
          MPI_IN_PLACE, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  // neighbor noncontiguous_complete_exchange
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoallv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_complete_exchange(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, true> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_alltoallw_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(send_buffer.datatype_first()),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.byte_displacement_first()),
          reinterpret_cast<MPI_Datatype const*>(receive_buffer.datatype_first()),
          topology.communicator().mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_complete_exchange", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif
