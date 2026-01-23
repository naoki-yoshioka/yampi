#ifndef YAMPI_NONCONTIGUOUS_SCATTER_HPP
# define YAMPI_NONCONTIGUOUS_SCATTER_HPP

# include <type_traits>
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
  // Blocking noncontiguous-scatter
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    using value_type = typename std::remove_cv<SendValue>::type;
    auto const error_code
      = MPI_Scatterv(
          const_cast<value_type*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
  }

  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::noncontiguous_scatter"};

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
  }

  template <typename Value>
  inline void noncontiguous_scatter(
    ::yampi::in_place_t const,
    ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_scatter"};

# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    using value_type = typename std::remove_cv<SendValue>::type;
    auto const error_code
      = MPI_Scatterv(
          const_cast<value_type*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
  }

  // only for intercommunicators
  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
  }

  template <typename SendValue>
  inline void noncontiguous_scatter(
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Scatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 3
    using value_type = typename std::remove_cv<SendValue>::type;
    auto const error_code
      = MPI_Scatterv(
          const_cast<value_type*>(send_buffer.data()), const_cast<int*>(send_buffer.count_first()),
          const_cast<int*>(send_buffer.displacement_first()), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
  }

  inline void noncontiguous_scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Scatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
  }

  // Nonblocking noncontiguous-scatter
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::immediate_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::noncontiguous_scatter"};

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_scatter(
    ::yampi::in_place_t const,
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_scatter"};

    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::immediate_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void noncontiguous_scatter(
    ::yampi::immediate_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv(
          send_buffer.data(), send_buffer.count_first(),
          send_buffer.displacement_first(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  inline void noncontiguous_scatter(
    ::yampi::immediate_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
# else // MPI_VERSION >= 4
    auto const error_code
      = MPI_Iscatterv(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), std::addressof(mpi_request));
# endif // MPI_VERSION >= 4
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }
# if MPI_VERSION >= 4

  // Persistent noncontiguous-scatter
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::noncontiguous_scatter"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_scatter(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root, ::yampi::information const& information,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_scatter"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request, ::yampi::information const& information,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), information.mpi_info(), std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  // information omitted
  // only for intracommunicators
  template <typename SendValue, typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error{"yampi::noncontiguous_scatter"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void noncontiguous_scatter(
    ::yampi::in_place_t const,
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<Value, false> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) != root)
      throw ::yampi::root_call_on_nonroot_error{"yampi::noncontiguous_scatter"};

    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, MPI_Count{0}, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  // only for intercommunicators
  template <typename ReceiveValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL,
          receive_buffer.data(), receive_buffer.count().mpi_count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::noncontiguous_buffer<SendValue, false> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          send_buffer.data(),
          reinterpret_cast<MPI_Count const*>(send_buffer.count_first()),
          reinterpret_cast<MPI_Aint const*>(send_buffer.displacement_first()),
          send_buffer.datatype().mpi_datatype(),
          nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }

  inline void noncontiguous_scatter(
    ::yampi::persistent_request& request,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    auto const error_code
      = MPI_Scatterv_init_c(
          nullptr, nullptr, nullptr, MPI_DATATYPE_NULL, nullptr, MPI_Count{0}, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm(), MPI_INFO_NULL, std::addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::noncontiguous_scatter", environment};
    request.reset(mpi_request, environment);
  }
# endif // MPI_VERSION >= 4
}


#endif
