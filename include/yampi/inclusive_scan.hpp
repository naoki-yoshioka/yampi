#ifndef YAMPI_INCLUSIVE_SCAN_HPP
# define YAMPI_INCLUSIVE_SCAN_HPP

# include <boost/config.hpp>

# include <cassert>
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
# include <yampi/binary_operation.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# include <yampi/in_place.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
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
  template <typename SendValue, typename ContiguousIterator>
  inline void inclusive_scan(
    ::yampi::buffer<SendValue> const& send_buffer,
    ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to SendValue");

    int const error_code
      = MPI_Scan(
          const_cast<SendValue*>(send_buffer.data()),
          const_cast<SendValue*>(YAMPI_addressof(*first)),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::inclusive_scan", environment);
  }

  template <typename SendValue>
  inline SendValue inclusive_scan(
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    assert(send_buffer.count() == 1);

    SendValue result;
    ::yampi::inclusive_scan(send_buffer, &result, operation, communicator, environment);
    return result;
  }

  template <typename Value>
  inline void inclusive_scan(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value>& receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Scan(
          MPI_IN_PLACE,
          receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::inclusive_scan", environment);
  }

  template <typename Value>
  inline void inclusive_scan(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> const& receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Scan(
          MPI_IN_PLACE,
          const_cast<Value*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::inclusive_scan", environment);
  }
# if MPI_VERSION >= 3


  template <typename SendValue, typename ContiguousIterator>
  inline void inclusive_scan(
    ::yampi::request& request,
    ::yampi::buffer<SendValue> const& send_buffer,
    ContiguousIterator const first,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    static_assert(
      (YAMPI_is_same<
         typename std::iterator_traits<ContiguousIterator>::value_type,
         SendValue>::value),
      "value_type of ContiguousIterator must be the same to SendValue");

    MPI_Request mpi_request;
    int const error_code
      = MPI_Iscan(
          const_cast<SendValue*>(send_buffer.data()),
          const_cast<SendValue*>(YAMPI_addressof(*first)),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::inclusive_scan", environment);

    request.reset(mpi_request, environment);
  }

  template <typename SendValue>
  inline SendValue inclusive_scan(
    ::yampi::request& request,
    ::yampi::buffer<SendValue> const& send_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    assert(send_buffer.count() == 1);

    SendValue result;
    ::yampi::inclusive_scan(request, send_buffer, &result, operation, communicator, environment);
    return result;
  }

  template <typename Value>
  inline void inclusive_scan(
    ::yampi::in_place_t const,
    ::yampi::request& request,
    ::yampi::buffer<Value>& receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Iscan(
          MPI_IN_PLACE,
          receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::inclusive_scan", environment);

    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void inclusive_scan(
    ::yampi::in_place_t const,
    ::yampi::request& request,
    ::yampi::buffer<Value> const& receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Iscan(
          MPI_IN_PLACE,
          const_cast<Value*>(receive_buffer.data()),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::inclusive_scan", environment);

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
