#ifndef YAMPI_NONBLOCKING_SEND_HPP
# define YAMPI_NONBLOCKING_SEND_HPP

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
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/request.hpp>
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
  inline void nonblocking_send(
    ::yampi::communicator const& communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::request& request, ::yampi::rank const destination, ::yampi::tag const tag)
  {
    MPI_Request request;
    int const error_code
      = MPI_Isend(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_send", environment);

    request.mpi_request(request);
  }


  namespace nonblocking_send_detail
  {
    template <typename CommunicationMode>
    struct nonblocking_send;

    template <>
    struct nonblocking_send< ::yampi::mode::standard_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer,
        ::yampi::request& request, ::yampi::rank const destination, ::yampi::tag const tag)
      { ::yampi::nonblocking_send(communicator, environment, buffer, request, destination, tag); }
    };

    template <>
    struct nonblocking_send< ::yampi::mode::buffered_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer,
        ::yampi::request& request, ::yampi::rank const destination, ::yampi::tag const tag)
      {
        MPI_Request mpi_request;
        int const error_code
          = MPI_Ibsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              YAMPI_addressof(mpi_request));
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::nonblocking_send", environment);

        request.mpi_request(mpi_request);
      }
    };

    template <>
    struct nonblocking_send< ::yampi::mode::synchronous_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer,
        ::yampi::request& request, ::yampi::rank const destination, ::yampi::tag const tag)
      {
        MPI_Request mpi_request;
        int const error_code
          = MPI_Issend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              YAMPI_addressof(mpi_request));
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::nonblocking_send", environment);

        request.mpi_request(mpi_request);
      }
    };

    template <>
    struct nonblocking_send< ::yampi::mode::ready_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::communicator const& communicator, ::yampi::environment const& environment,
        ::yampi::buffer<Value> const& buffer,
        ::yampi::request& request, ::yampi::rank const destination, ::yampi::tag const tag)
      {
        MPI_Request mpi_request;
        int const error_code
          = MPI_Irsend(
              const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
              YAMPI_addressof(mpi_request));
        if (error_code != MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::nonblocking_send", environment);

        request.mpi_request(mpi_request);
      }
    };
  }

  template <typename CommunicationMode, typename Value>
  inline void nonblocking_send(
    YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::request& request, ::yampi::rank const destination, ::yampi::tag const tag)
  {
    typedef typename YAMPI_remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::nonblocking_send_detail::nonblocking_send<communication_mode_type>::call(
      YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
      communicator, environment, buffer, request, destination, tag);
  }
}


# undef YAMPI_remove_reference
# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY
# undef YAMPI_addressof

#endif

