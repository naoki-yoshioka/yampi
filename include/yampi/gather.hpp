#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

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
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>

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
  // TODO: implement MPI_Gatherv
  template <typename SendValue, typename ContiguousIterator>
  inline void gather(
    ::yampi::communicator const& communicator, ::yampi::rank const root,
    ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ContiguousIterator const first)
  {
    static_assert(
      (YAMPI_is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to SendValue");

    int const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<SendValue*>(YAMPI_addressof(*first)),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void gather(
    ::yampi::communicator const& communicator, ::yampi::rank const root,
    ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue>& receive_buffer)
  {
    int const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          YAMPI_addressof(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void gather(
    ::yampi::communicator const& communicator, ::yampi::rank const root,
    ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::buffer<ReceiveValue> const& receive_buffer)
  {
    int const error_code
      = MPI_Gather(
          const_cast<SendValue*>(send_buffer.data()),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          const_cast<ReceiveValue*>(YAMPI_addressof(receive_buffer.data())),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::gather", environment);
  }

  template <typename SendValue>
  inline void gather(
    ::yampi::communicator const& communicator, ::yampi::rank const root,
    ::yampi::environment const& environment,
    ::yampi::buffer<SendValue> const& send_buffer)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::gather");

    SendValue null;
    ::yampi::gather(communicator, root, environment, send_buffer, YAMPI_addressof(null));
  }
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_same

#endif

