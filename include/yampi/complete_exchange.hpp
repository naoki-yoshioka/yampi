#ifndef YAMPI_COMPLETE_EXCHANGE_HPP
# define YAMPI_COMPLETE_EXCHANGE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
# endif
# include <iterator>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
#   include <yampi/topology.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
# else
#   define YAMPI_is_same boost::is_same
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert BOOST_STATIC_ASSERT_MSG
# endif


namespace yampi
{
  // TODO: implement MPI_Alltoallv, MPI_Alltoallw
  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Alltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }
# if MPI_VERSION >= 3


  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::request& request,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Ialltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);

    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::request& request,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Ialltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);

    request.reset(mpi_request, environment);
  }


  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::topology const& topology,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Neighbor_alltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::topology const& topology,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Neighbor_alltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);
  }


  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::request& request,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue>& receive_buffer,
    ::yampi::topology const& topology,
    ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Ineighor_alltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);

    request.reset(mpi_request, environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void complete_exchange(
    ::yampi::request& request,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue> const& receive_buffer,
    ::yampi::topology const& topology,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Ineighor_alltoall(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<ReceiveValue*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          topology.communicator().mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::complete_exchange", environment);

    request.reset(mpi_request, environment);
  }
# endif
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_same

#endif
