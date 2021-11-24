#ifndef YAMPI_REDUCE_HPP
# define YAMPI_REDUCE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_same.hpp>
#   include <boost/type_traits/has_nothrow_copy.hpp>
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
# include <yampi/rank.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/in_place.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request_base.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# include <yampi/root_call_on_nonroot_error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_same std::is_same
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_same boost::is_same
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
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
  class reduce
  {
    ::yampi::rank root_;
    ::yampi::communicator const& communicator_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    reduce() = delete;
    reduce(reduce const&) = delete;
    reduce& operator=(reduce const&) = delete;
# else
   private:
    reduce();
    reduce(reduce const&);
    reduce& operator=(reduce const&);

   public:
# endif

    reduce(
      ::yampi::rank const& root, ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value)
      : root_(root), communicator_(communicator)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    reduce(reduce&&) = default;
    reduce& operator=(reduce&&) = default;
#   endif
    ~reduce() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    template <typename SendValue, typename ContiguousIterator>
    void call(
      ::yampi::buffer<SendValue> const send_buffer,
      ContiguousIterator const first,
      ::yampi::binary_operation const& operation,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

# if MPI_VERSION >= 3
      int const error_code
        = MPI_Reduce(
            send_buffer.data(), YAMPI_addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root_.mpi_rank(), communicator_.mpi_comm());
# else //MPI_VERSION >= 3
      int const error_code
        = MPI_Reduce(
            const_cast<SendValue*>(send_buffer.data()), YAMPI_addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root_.mpi_rank(), communicator_.mpi_comm());
# endif //MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::reduce::call", environment);
    }

    template <typename SendValue>
    void call(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::reduce::call");

      SendValue null;
      call(send_buffer, YAMPI_addressof(null), operation, environment);
    }

    template <typename Value>
    void call(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) != root_)
        throw ::yampi::root_call_on_nonroot_error("yampi::reduce::call");

      int const error_code
        = MPI_Reduce(
            MPI_IN_PLACE, receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::reduce::call", environment);
    }
  };

# if MPI_VERSION >= 3
  class reduce_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    reduce_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    reduce_request(reduce_request const&) = default;
    reduce_request& operator=(reduce_request const&) = default;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    reduce_request(reduce_request&&) = default;
    reduce_request& operator=(reduce_request&&) = default;
#     endif
    ~reduce_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    template <typename SendValue, typename ContiguousIterator>
    reduce_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_reduce_request(send_buffer, first, operation, root, communicator, environment))
    { }

    template <typename SendValue>
    reduce_request(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_reduce_request(send_buffer, operation, root, communicator, environment))
    { }

    template <typename Value>
    reduce_request(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_reduce_in_place_request(receive_buffer, operation, root, communicator, environment))
    { }

   private:
    // Nonblocking reduce
    template <typename SendValue, typename ContiguousIterator>
    static void do_reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      int const error_code
        = MPI_Ireduce(
            send_buffer.data(), YAMPI_addressof(*first),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::reduce_request::do_reduce", environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    static MPI_Request make_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_reduce(result, send_buffer, first, operation, root, communicator, environment);
      return result;
    }

    template <typename SendValue>
    static void do_reduce(
      MPI_Request& mpi_request,
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) == root)
        throw ::yampi::nonroot_call_on_root_error("yampi::reduce_request::do_reduce");

      SendValue null;
      do_reduce(mpi_request, send_buffer, YAMPI_addressof(null), operation, root, communicator, environment);
    }

    template <typename SendValue>
    static MPI_Request make_reduce_request(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_reduce(result, send_buffer, operation, root, communicator, environment);
      return result;
    }

    template <typename Value>
    static void do_reduce_in_place(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      if (communicator.rank(environment) != root)
        throw ::yampi::root_call_on_nonroot_error("yampi::reduce_request::do_reduce_in_place");

      int const error_code
        = MPI_Ireduce(
            MPI_IN_PLACE, receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root.mpi_rank(), communicator.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::reduce_request::do_reduce_in_place", environment);
    }

    template <typename Value>
    static MPI_Request make_reduce_in_place_request(
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_reduce_in_place(result, receive_buffer, operation, root, communicator, environment);
      return result;
    }

   public:
    template <typename SendValue, typename ContiguousIterator>
    void reset(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, first, operation, root, communicator, environment);
    }

    template <typename SendValue>
    void reset(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(send_buffer, operation, root, communicator, environment);
    }

    template <typename Value>
    void reset(
      ::yampi::in_place_t const in_place,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      reduce(in_place, receive_buffer, operation, root, communicator, environment);
    }

    template <typename SendValue, typename ContiguousIterator>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer, ContiguousIterator const first,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_reduce(mpi_request_, send_buffer, first, operation, root, communicator, environment); }

    template <typename SendValue>
    void reduce(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_reduce(mpi_request_, send_buffer, operation, root, communicator, environment); }

    template <typename Value>
    void reduce(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> receive_buffer,
      ::yampi::binary_operation const& operation, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_reduce_in_place(mpi_request_, receive_buffer, operation, root, communicator, environment); }
  };
# endif // MPI_VERSION >= 3
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_same

#endif
