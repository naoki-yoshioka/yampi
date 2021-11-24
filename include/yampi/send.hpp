#ifndef YAMPI_SEND_HPP
# define YAMPI_SEND_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   include <utility>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/remove_reference.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# include <yampi/communication_mode.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(T) T&&
#   define YAMPI_FORWARD_OR_COPY(T, x) std::forward<T>(x)
# else
#   define YAMPI_RVALUE_REFERENCE_OR_COPY(T) T
#   define YAMPI_FORWARD_OR_COPY(T, x) x
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_remove_reference std::remove_reference
# else
#   define YAMPI_remove_reference boost::remove_reference
# endif


namespace yampi
{
  template <typename Value>
  inline void send(
    ::yampi::buffer<Value> const buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
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
    struct send< ::yampi::mode::standard_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::buffer<Value> const buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      { ::yampi::send(communicator, environment, buffer, destination, tag); }
    };

    template <>
    struct send< ::yampi::mode::buffered_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::buffer<Value> const buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
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
    struct send< ::yampi::mode::synchronous_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::buffer<Value> const buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
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
    struct send< ::yampi::mode::ready_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::buffer<Value> const buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
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
    YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
    ::yampi::buffer<Value> const buffer, ::yampi::rank const& destination, ::yampi::tag const& tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    typedef typename YAMPI_remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::send_detail::send<communication_mode_type>::call(
      YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
      buffer, destination, tag, communicator, environment);
  }
}


# undef YAMPI_remove_reference
# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY
# undef YAMPI_addressof

#endif

