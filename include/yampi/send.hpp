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
# include <yampi/immediate_request.hpp>
# include <yampi/persistent_request.hpp>


namespace yampi
{
  // Blocking send
  template <typename Value>
  inline void send(
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Send_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Send(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION
    auto const error_code
      = MPI_Send(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send", environment};
  }

  // Nonblocking send
  template <typename Value>
  inline void send(
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Isend_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Isend(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION
    auto const error_code
      = MPI_Isend(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send", environment};
    request.reset(mpi_request, environment);
  }

  // Persistent send
  template <typename Value>
  inline void send(
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
# if MPI_VERSION >= 4
    auto const error_code
      = MPI_Send_init_c(
          buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# elif MPI_VERSION >= 3
    auto const error_code
      = MPI_Send_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# else // MPI_VERSION >= 3
    auto const error_code
      = MPI_Send_init(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error{error_code, "yampi::send", environment};
    request.reset(mpi_request, environment);
  }

  namespace send_detail
  {
    template <typename CommunicationMode>
    struct send;

    template <>
    struct send< ::yampi::mode::standard_communication_t >
    {
      template <typename Value>
      static void call(
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      { ::yampi::send(buffer, destination, tag, communicator, environment); }

      template <typename Value>
      static void call(
        ::yampi::immediate_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      { ::yampi::send(request, buffer, destination, tag, communicator, environment); }

      template <typename Value>
      static void call(
        ::yampi::persistent_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      { ::yampi::send(request, buffer, destination, tag, communicator, environment); }
    };

    template <>
    struct send< ::yampi::mode::buffered_communication_t >
    {
      // Blocking buffered-send
      template <typename Value>
      static void call(
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Bsend_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Bsend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION
        auto const error_code
          = MPI_Bsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
      }

      // Nonblocking buffered-send
      template <typename Value>
      static void call(
        ::yampi::immediate_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Ibsend_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Ibsend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# else // MPI_VERSION
        auto const error_code
          = MPI_Ibsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# endif // MPI_VERSION
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
        request.reset(mpi_request, environment);
      }

      // Persistent buffered-send
      template <typename Value>
      static void call(
        ::yampi::persistent_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Bsend_init_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Bsend_init(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# else // MPI_VERSION >= 3
        auto const error_code
          = MPI_Bsend_init(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
        request.reset(mpi_request, environment);
      }
    };

    template <>
    struct send< ::yampi::mode::synchronous_communication_t >
    {
      // Blocking synchronous-send
      template <typename Value>
      static void call(
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Ssend_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Ssend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION
        auto const error_code
          = MPI_Ssend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
      }

      // Nonblocking synchronous-send
      template <typename Value>
      static void call(
        ::yampi::immediate_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Issend_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Issend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# else // MPI_VERSION
        auto const error_code
          = MPI_Issend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# endif // MPI_VERSION
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
        request.reset(mpi_request, environment);
      }

      // Persistent synchronous-send
      template <typename Value>
      static void call(
        ::yampi::persistent_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Ssend_init_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Ssend_init(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# else // MPI_VERSION >= 3
        auto const error_code
          = MPI_Ssend_init(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
        request.reset(mpi_request, environment);
      }
    };

    template <>
    struct send< ::yampi::mode::ready_communication_t >
    {
      // Blocking ready-send
      template <typename Value>
      static void call(
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Rsend_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Rsend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# else // MPI_VERSION
        auto const error_code
          = MPI_Rsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
# endif // MPI_VERSION
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
      }

      // Nonblocking ready-send
      template <typename Value>
      static void call(
        ::yampi::immediate_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Irsend_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Irsend(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# else // MPI_VERSION
        auto const error_code
          = MPI_Irsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# endif // MPI_VERSION
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
        request.reset(mpi_request, environment);
      }

      // Persistent ready-send
      template <typename Value>
      static void call(
        ::yampi::persistent_request& request,
        ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
# if MPI_VERSION >= 4
        auto const error_code
          = MPI_Rsend_init_c(
              buffer.data(), buffer.count().mpi_count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# elif MPI_VERSION >= 3
        auto const error_code
          = MPI_Rsend_init(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# else // MPI_VERSION >= 3
        auto const error_code
          = MPI_Rsend_init(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              std::addressof(mpi_request));
# endif // MPI_VERSION >= 3
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error{error_code, "yampi::send", environment};
        request.reset(mpi_request, environment);
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
      buffer, destination, tag, communicator, environment);
  }

  template <typename CommunicationMode, typename Value>
  inline void send(
    CommunicationMode&& communication_mode,
    ::yampi::immediate_request& request,
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    typedef typename std::remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::send_detail::send<communication_mode_type>::call(
      request, buffer, destination, tag, communicator, environment);
  }

  template <typename CommunicationMode, typename Value>
  inline void send(
    CommunicationMode&& communication_mode,
    ::yampi::persistent_request& request,
    ::yampi::buffer<Value> const buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
    typedef typename std::remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::send_detail::send<communication_mode_type>::call(
      request, buffer, destination, tag, communicator, environment);
  }
}


#endif

