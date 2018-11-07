#ifndef YAMPI_INITIALIZE_SEND_HPP
# define YAMPI_INITIALIZE_SEND_HPP

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
  inline void initialize_send(
    ::yampi::request& request,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Send_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code == MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::initialize_send", environment);

    request.reset(mpi_request, environment);
  }


  namespace initialize_send_detail
  {
    template <typename CommunicationMode>
    struct initialize_send;

    template <>
    struct initialize_send< ::yampi::mode::standard_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::request& request,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      { initialize(request, buffer, destination, tag, communicator, environment); }
    };

    template <>
    struct initialize_send< ::yampi::mode::buffered_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::request& request,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
        int const error_code
          = MPI_Bsend_init(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
        if (error_code == MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::initialize_send", environment);

        request.reset(mpi_request, environment);
      }
    };

    template <>
    struct initialize_send< ::yampi::mode::synchronous_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::request& request,
        ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
        int const error_code
          = MPI_Ssend_init(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
        if (error_code == MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::initialize_send", environment);

        request.reset(mpi_request, environment);
      }
    };

    template <>
    struct initialize_send< ::yampi::mode::ready_communication >
    {
      template <typename CommunicationMode, typename Value>
      static void call(
        YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode),
        ::yampi::request& request,
        ::yampi::buffer<Value> const& send_buffer, ::yampi::rank const destination, ::yampi::tag const send_tag,
        ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      {
        MPI_Request mpi_request;
        int const error_code
          = MPI_Rsend_init(
              buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
              destination.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
        if (error_code == MPI_SUCCESS)
          throw ::yampi::error(error_code, "yampi::initialize_send", environment);

        request.reset(mpi_request, environment);
      }
    };
  } // namespace initialize_send_detail


  template <typename CommunicationMode, typename Value>
  inline void initialize_send(
    YAMPI_RVALUE_REFERENCE_OR_COPY(CommunicationMode) communication_mode,
    ::yampi::request& request,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const destination, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    typedef typename YAMPI_remove_reference<CommunicationMode>::type communication_mode_type;
    ::yampi::initialize_send_detail::initialize_send<communication_mode_type>::call(
      YAMPI_FORWARD_OR_COPY(CommunicationMode, communication_mode),
      request, buffer, destination, tag, communicator, environment);
  }
}


# undef YAMPI_remove_reference
# undef YAMPI_FORWARD_OR_COPY
# undef YAMPI_RVALUE_REFERENCE_OR_COPY
# undef YAMPI_addressof

#endif

