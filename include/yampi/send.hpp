#ifndef YAMPI_SEND_HPP
# define YAMPI_SEND_HPP

# include <memory>
# include <utility>
# include <type_traits>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# include <yampi/communication_mode.hpp>


namespace yampi
{
  template <typename Value>
  inline void send(
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    int const error_code
      = MPI_Send(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Send(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::send", environment);
  }

  namespace send_detail
  {
    template <typename CommunicationMode>
    struct send;

    template <>
    struct send< ::yampi::mode::standard_communication_t >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        CommunicationMode&&,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      { ::yampi::send(communicator, environment, buffer, destination, tag); }
    };

    template <>
    struct send< ::yampi::mode::buffered_communication_t >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        CommunicationMode&&,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
# if MPI_VERSION >= 3
        int const error_code
          = MPI_Bsend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
        int const error_code
          = MPI_Bsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::send", environment);
      }
    };

    template <>
    struct send< ::yampi::mode::synchronous_communication_t >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        CommunicationMode&&,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
# if MPI_VERSION >= 3
        int const error_code
          = MPI_Ssend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
        int const error_code
          = MPI_Ssend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::send", environment);
      }
    };

    template <>
    struct send< ::yampi::mode::ready_communication_t >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        CommunicationMode&&,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
# if MPI_VERSION >= 3
        int const error_code
          = MPI_Rsend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
        int const error_code
          = MPI_Rsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::send", environment);
      }
    };
  }


  template <typename CommunicationMode, typename Value>
  inline void send(
    CommunicationMode&& communication_mode,
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    typedef typename std::remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::send_detail::send<communication_mode_type>::call(
      std::forward<CommunicationMode>(communication_mode),
      buffer, destination, tag, communicator, environment);
  }
}


#endif

