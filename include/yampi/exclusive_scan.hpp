#ifndef YAMPI_EXCLUSIVE_SCAN_HPP
# define YAMPI_EXCLUSIVE_SCAN_HPP

# include <boost/config.hpp>

# include <cassert>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
#   if MPI_VERSION >= 3
#     include <boost/type_traits/has_nothrow_copy.hpp>
#   endif // MPI_VERSION >= 3
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

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/rank.hpp>
# include <yampi/in_place.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request_base.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   if MPI_VERSION >= 3
#     define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   endif // MPI_VERSION >= 3
# else
#   define YAMPI_is_same boost::is_same
#   if MPI_VERSION >= 3
#     define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   endif // MPI_VERSION >= 3
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
  inline void exclusive_scan(
    ::yampi::buffer<SendValue> const send_buffer,
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

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Exscan(
          send_buffer.data(), YAMPI_addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Exscan(
          const_cast<SendValue*>(send_buffer.data()), YAMPI_addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::exclusive_scan", environment);
  }

  // only for blocking exclusive_scan
  template <typename SendValue>
  inline SendValue exclusive_scan(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    assert(send_buffer.count() == 1);

    SendValue result;
    ::yampi::exclusive_scan(send_buffer, &result, operation, communicator, environment);
    return result;
  }

  template <typename Value>
  inline void exclusive_scan(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Exscan(
          MPI_IN_PLACE, receive_buffer.data(),
          receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::exclusive_scan", environment);
  }
# if MPI_VERSION >= 3

  class exclusive_scan_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    exclusive_scan_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    exclusive_scan_request(exclusive_scan_request const&) = default;
    exclusive_scan_request& operator=(exclusive_scan_request const&) = default;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    exclusive_scan_request(exclusive_scan_request&&) = default;
    exclusive_scan_request& operator=(exclusive_scan_request&&) = default;
#     endif
    ~exclusive_scan_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    template <typename SendValue, typename ContiguousIterator>
    exclusive_scan_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_exclusive_scan_request(send_buffer, first, operation, communicator, environment))
    { }

    template <typename Value>
    exclusive_scan_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_exclusive_scan_in_place_request(receive_buffer, operation, communicator, environment))
    { }

   private:
    template <typename SendValue, typename ContiguousIterator>
    static void do_exclusive_scan(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      int const error_code
        = MPI_Iexscan(
            send_buffer.data(), YAMPI_addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exclusive_scan_request::do_exclusive_scan", environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    static MPI_Request make_exclusive_scan_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_exclusive_scan(result, send_buffer, first, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_exclusive_scan_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) != root)
        throw ::yampi::root_call_on_nonroot_error("yampi::exclusive_scan_request::do_exclusive_scan_in_place");

      int const error_code
        = MPI_Iexscan(
            MPI_IN_PLACE, receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exclusive_scan_request::do_exclusive_scan_in_place", environment);
    }

    template <typename Value>
    static MPI_Request make_exclusive_scan_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_exclusive_scan_in_place(result, receive_buffer, operation, communication, environment);
      return result;
    }

   public:
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      exclusive_scan(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      exclusive_scan(in_place, receive_buffer, operation, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void exclusive_scan(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_exclusive_scan(mpi_request_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void exclusive_scan(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_exclusive_scan_in_place(mpi_request_, receive_buffer, operation, communicator, environment); }
  };
# endif // MPI_VERSION >= 3
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# if MPI_VERSION >= 3
#   undef YAMPI_is_nothrow_copy_constructible
# endif
# undef YAMPI_is_same

#endif
