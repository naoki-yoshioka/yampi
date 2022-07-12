#ifndef YAMPI_SCATTER_HPP
# define YAMPI_SCATTER_HPP

# include <cassert>
# include <type_traits>
# include <iterator>
# include <memory>

# include <mpi.h>

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>


namespace yampi
{
  // TODO: implement MPI_Scatterv
  template <typename ContiguousIterator, typename ReceiveValue>
  inline void scatter(
    ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    static_assert(
      (std::is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         ReceiveValue>::value),
      "value_type of ContiguousIterator must be the same to ReceiveValue");
    assert(std::addressof(*first) != receive_buffer.data());

    int const error_code
      = MPI_Scatter(
          std::addressof(*first), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void scatter(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    assert(send_buffer.data() != receive_buffer.data());

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename ReceiveValue>
  inline void scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::scatter");

    int const error_code
      = MPI_Scatter(
          nullptr, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename ReceiveValue>
  inline void scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Scatter(
          nullptr, receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename Value>
  inline void scatter(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::root_call_on_nonroot_error("yampi::scatter");

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          const_cast<Value*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename SendValue>
  inline void scatter(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  inline void scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Scatter(
          nullptr, 0, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }
}


#endif
