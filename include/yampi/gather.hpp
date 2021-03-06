#ifndef YAMPI_GATHER_HPP
# define YAMPI_GATHER_HPP

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

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>
# include <yampi/nonroot_call_on_root_error.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
# endif

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
  // TODO: implement MPI_Gatherv
  class gather
  {
    ::yampi::rank root_;
    ::yampi::communicator const& communicator_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    gather() = delete;
    gather(gather const&) = delete;
    gather& operator=(gather const&) = delete;
# else
   private:
    gather();
    gather(gather const&);
    gather& operator=(gather const&);

   public:
# endif

    gather(
      ::yampi::rank const& root, ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value)
      : root_(root), communicator_(communicator)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    gather(gather&&) = default;
    gather& operator=(gather&&) = default;
#   endif
    ~gather() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename SendValue, typename ContiguousIterator>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ContiguousIterator const first,
      ::yampi::environment const& environment) const
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
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue>& receive_buffer,
      ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Gather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue> const& receive_buffer,
      ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Gather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<ReceiveValue*>(receive_buffer.data()),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);
    }

    template <typename SendValue>
    void call(
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");

      SendValue null;
      call(send_buffer, YAMPI_addressof(null), environment);
    }
# if MPI_VERSION >= 3


    template <typename SendValue, typename ContiguousIterator>
    void call(
      ::yampi::request& request,
      ::yampi::buffer<SendValue> const& send_buffer,
      ContiguousIterator const first,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           SendValue>::value),
        "value_type of ContiguousIterator must be the same to SendValue");

      MPI_Request mpi_request;
      int const error_code
        = MPI_Igather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<SendValue*>(YAMPI_addressof(*first)),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);

      request.reset(mpi_request, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::request& request,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue>& receive_buffer,
      ::yampi::environment const& environment) const
    {
      MPI_Request mpi_request;
      int const error_code
        = MPI_Igather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);

      request.reset(mpi_request, environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::request& request,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::buffer<ReceiveValue> const& receive_buffer,
      ::yampi::environment const& environment) const
    {
      MPI_Request mpi_request;
      int const error_code
        = MPI_Igather(
            const_cast<SendValue*>(send_buffer.data()),
            send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            const_cast<ReceiveValue*>(receive_buffer.data()),
            receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::gather::call", environment);

      request.reset(mpi_request, environment);
    }

    template <typename SendValue>
    void call(
      ::yampi::request& request,
      ::yampi::buffer<SendValue> const& send_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::gather::call");

      SendValue null;
      call(request, send_buffer, YAMPI_addressof(null), environment);
    }
# endif
  };
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_same

#endif

