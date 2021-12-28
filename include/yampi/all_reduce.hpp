#ifndef YAMPI_ALL_REDUCE_HPP
# define YAMPI_ALL_REDUCE_HPP

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
  inline void all_reduce(
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
      = MPI_Allreduce(
          send_buffer.data(), YAMPI_addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Allreduce(
          const_cast<SendValue*>(send_buffer.data()), YAMPI_addressof(*first),
          send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::all_reduce", environment);
  }

  // only for blocking all_reduce
  template <typename SendValue>
  inline SendValue all_reduce(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    assert(send_buffer.count() == 1);

    SendValue result;
    ::yampi::all_reduce(send_buffer, &result, operation, communicator, environment);
    return result;
  }

  template <typename Value>
  inline void all_reduce(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> receive_buffer,
    ::yampi::binary_operation const& operation,
    ::yampi::communicator const& communicator,
    ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Allreduce(
          MPI_IN_PLACE,
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          operation.mpi_op(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::all_reduce", environment);
  }
# if MPI_VERSION >= 3

  class all_reduce_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    all_reduce_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    all_reduce_request(all_reduce_request const&) = delete;
    all_reduce_request& operator=(all_reduce_request const&) = delete;
#   else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    all_reduce_request(all_reduce_request const&);
    all_reduce_request& operator=(all_reduce_request const&);

   public:
#   endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    all_reduce_request(all_reduce_request&&) = default;
    all_reduce_request& operator=(all_reduce_request&&) = default;
#     endif
    ~all_reduce_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    template <typename SendValue, typename ContiguousIterator>
    all_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_all_reduce_request(send_buffer, first, operation, communicator, environment))
    { }

    template <typename Value>
    all_reduce_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_all_reduce_in_place_request(receive_buffer, operation, communicator, environment))
    { }

   private:
    template <typename SendValue, typename ContiguousIterator>
    static void do_all_reduce(
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
        = MPI_Iallreduce(
            send_buffer.data(), YAMPI_addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::all_reduce_request::do_all_reduce", environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    static MPI_Request make_all_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_all_reduce(result, send_buffer, first, operation, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_all_reduce_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Iallreduce(
            MPI_IN_PLACE,
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::all_reduce_request::do_all_reduce_in_place", environment);
    }

    template <typename Value>
    static MPI_Request make_all_reduce_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_all_reduce_in_place(result, receive_buffer, operation, communicator, environment);
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
      all_reduce(send_buffer, first, operation, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      all_reduce(in_place, receive_buffer, operation, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void all_reduce(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_all_reduce(mpi_request_, send_buffer, first, operation, communicator, environment); }

    template <typename Value>
    void all_reduce(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_all_reduce_in_place(mpi_request_, receive_buffer, operation, communicator, environment); }
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
