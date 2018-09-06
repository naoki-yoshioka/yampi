#ifndef YAMPI_REDUCE_HPP
# define YAMPI_REDUCE_HPP

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
# include <yampi/binary_operation.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
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
  class reduce
  {
    ::yampi::communicator const& communicator_;
    ::yampi::rank root_;

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
      ::yampi::communicator const& communicator, ::yampi::rank const root)
      BOOST_NOEXCEPT_OR_NOTHROW
      : communicator_(communicator), root_(root)
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
      ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ContiguousIterator const first,
      ::yampi::binary_operation const& operation) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      int const error_code
        = MPI_Reduce(
            const_cast<SendValue*>(send_buffer.data()),
            const_cast<SendValue*>(YAMPI_addressof(*first)),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::reduce::call", environment);
    }

    template <typename SendValue>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::binary_operation const& operation) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::reduce::call");

      SendValue null;
      call(environment, send_buffer, YAMPI_addressof(null), operation);
    }
# if MPI_VERSION >= 3


    template <typename SendValue, typename ContiguousIterator>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ContiguousIterator const first,
      ::yampi::request& request,
      ::yampi::binary_operation const& operation) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      MPI_Request mpi_request;
      int const error_code
        = MPI_Ireduce(
            const_cast<SendValue*>(send_buffer.data()),
            const_cast<SendValue*>(YAMPI_addressof(*first)),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            operation.mpi_op(), root_.mpi_rank(), communicator_.mpi_comm(),
            YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::reduce::call", environment);

      request.release(environment);
      request.mpi_request(mpi_request);
    }

    template <typename SendValue>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::request& request,
      ::yampi::binary_operation const& operation) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::reduce::call");

      SendValue null;
      call(environment, send_buffer, YAMPI_addressof(null), request, operation);
    }
# endif
  };
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_same

#endif
