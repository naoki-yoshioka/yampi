#ifndef YAMPI_BLOCKING_SEND_HPP
# define YAMPI_BLOCKING_SEND_HPP

# include <boost/config.hpp>

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
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# include <yampi/communication_mode.hpp>

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
  inline void blocking_send(
    ::yampi::communicator const& communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
  {
    int const error_code
      = MPI_Send(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::blocking_send", environment);
  }


  namespace blocking_send_detail
  {
    template <typename CommunicationMode>
    struct blocking_send;

    template <>
    struct blocking_send< ::yampi::mode::standard_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
      { ::yampi::blocking_send(communicator, environment, buffer, destination, tag); }
    };

    template <>
    struct blocking_send< ::yampi::mode::buffered_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
      {
        int const error_code
          = MPI_Bsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::blocking_send", environment);
      }
    };

    template <>
    struct blocking_send< ::yampi::mode::synchronous_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
      {
        int const error_code
          = MPI_Ssend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::blocking_send", environment);
      }
    };

    template <>
    struct blocking_send< ::yampi::mode::ready_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
      {
        int const error_code
          = MPI_Rsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm());
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::blocking_send", environment);
      }
    };
  }


  template <typename CommunicationMode, typename Value>
  inline void blocking_send(
    YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag)
  {
    typedef typename YAMPI_remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::blocking_send_detail::blocking_send<communication_mode_type>::call(
      YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
      communicator, environment, buffer, destination, tag);
  }
}


# undef YAMPI_remove_reference
# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY

#endif

