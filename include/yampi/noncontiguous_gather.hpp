#ifndef YAMPI_NONCONTIGUOUS_GATHER_HPP
# define YAMPI_NONCONTIGUOUS_GATHER_HPP

# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/noncontiguous_buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>
# include <yampi/immediate_request.hpp>
# if MPI_VERSION >= 4
#   include <yampi/persistent_request.hpp>
#   include <yampi/information.hpp>
# endif // MPI_VERSION >= 4


namespace yampi
{
  // Blocking noncontiguous-gather
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Gatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gatherv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
  }

  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::noncontiguos_gather");

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Gatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gatherv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
  }

  template <typename Value>
  inline void noncontiguous_gather(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_gather"};

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Gatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Gatherv(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv(
          nullptr, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Gatherv(
          nullptr, 0, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
  }

  // Nonblocking noncontiguous-gather
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::noncontiguos_gather"};

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_gather(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_gather"};

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv(
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::immediate_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv(
          nullptr, 0, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count_first(),
          receive_buffer.displacement_first(), receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::immediate_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Igatherv(
          nullptr, 0, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 4

  // Persistent noncontiguous-gather
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::noncontiguos_gather");

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_gather(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_gather"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::noncontiguos_gather");

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_gather(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_gather"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename SendValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          send_buffer.data(), send_buffer.count().mpi_count(), send_buffer.datatype().mpi_datatype(),
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<ReceiveValue, false> receive_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          receive_buffer.data(),
          reinterpret_cast<MPI_Count const*>(receive_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(receive_buffer.displacement_first()),
          receive_buffer.datatype().mpi_datatype(),
          MPI_ROOT, communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_gather(
    ::yampi::persistent_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Gatherv_init_c(
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL, nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_gather", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif

