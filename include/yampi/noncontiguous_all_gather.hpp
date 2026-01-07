#ifndef YAMPI_NONCONTIGUOUS_ALL_GATHER_HPP
# define YAMPI_NONCONTIGUOUS_ALL_GATHER_HPP

# include <type_traits>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
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
  // Blocking noncontiguous-all-gather
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_all_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Allgatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Allgatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 3
    using value_type = typename std::remove_cv<SendValue>::type;
    auto const error_code
      = MPI_Allgatherv(
          const_cast<value_type*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_all_gather(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Allgatherv_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Allgatherv(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
  }
# if MPI_VERSION >= 3

  // neighbor noncontiguous_all_gather
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_all_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_allgatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Neighbor_allgatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
  }
# endif // MPI_VERSION >= 3

  // Nonblocking noncontiguous-all-gather
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_all_gather(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallgatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallgatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_all_gather(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallgatherv_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iallgatherv(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 3

  // neighbor noncontiguous_all_gather
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_all_gather(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_allgatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Ineighbor_allgatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 3
# if MPI_VERSION >= 4

  // Persistent noncontiguous-all-gather
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_all_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::information const& information,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allgatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_all_gather(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allgatherv_init_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }

  // neighbor noncontiguous_all_gather
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_all_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue> receive_buffer,
    ::yampi::information const& information,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_allgatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_all_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allgatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }

  // only for intracommunicators
  template <typename Value>
  inline void noncontiguous_all_gather(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Allgatherv_init_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }

  // neighbor noncontiguous_all_gather
  template <typename SendValue, typename ReceiveValue, typename Topology>
  inline void noncontiguous_all_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue> receive_buffer,
    ::yampi::topology<Topology> const& topology, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Neighbor_allgatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_all_gather", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif
