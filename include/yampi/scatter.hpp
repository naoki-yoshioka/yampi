#ifndef YAMPI_SCATTER_HPP
# define YAMPI_SCATTER_HPP

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
# include <yampi/in_place.hpp>
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
  // TODO: implement MPI_Scatterv
  class scatter
  {
    ::yampi::rank root_;
    ::yampi::communicator const& communicator_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    scatter() = delete;
    scatter(scatter const&) = delete;
    scatter& operator=(scatter const&) = delete;
# else
   private:
    scatter();
    scatter(scatter const&);
    scatter& operator=(scatter const&);

   public:
# endif

    scatter(
      ::yampi::rank const& root, ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value)
      : root_(root), communicator_(communicator)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    scatter(scatter&&) = default;
    scatter& operator=(scatter&&) = default;
#   endif
    ~scatter() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename ContiguousIterator, typename ReceiveValue>
    void call(
      ContiguousIterator const first,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::environment const& environment) const
    {
      static_assert(
        (YAMPI_is_same<
           typename std::iterator_traits<ContiguousIterator>::value_type,
           ReceiveValue>::value),
        "value_type of ContiguousIterator must be the same to ReceiveValue");

      int const error_code
        = MPI_Scatter(
            YAMPI_addressof(*first), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::scatter::call", environment);
    }

    template <typename SendValue, typename ReceiveValue>
    void call(
      ::yampi::buffer<SendValue> const send_buffer,
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::environment const& environment) const
    {
# if MPI_VERSION >= 3
      int const error_code
        = MPI_Scatter(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Scatter(
            const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::scatter::call", environment);
    }

    template <typename ReceiveValue>
    void call(
      ::yampi::buffer<ReceiveValue> receive_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::nonroot_call_on_root_error("yampi::scatter::call");

      ReceiveValue null;
      call(YAMPI_addressof(null), receive_buffer, environment);
    }

    template <typename Value>
    void call(
      ::yampi::in_place_t const,
      ::yampi::buffer<Value> const send_buffer,
      ::yampi::environment const& environment) const
    {
      if (communicator_.rank(environment) == root_)
        throw ::yampi::root_call_on_nonroot_error("yampi::scatter::call");

# if MPI_VERSION >= 3
      int const error_code
        = MPI_Scatter(
            send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            MPI_IN_PLACE, send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
# else // MPI_VERSION >= 3
      int const error_code
        = MPI_Scatter(
            const_cast<Value*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            MPI_IN_PLACE, send_buffer.count(), send_buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
# endif // MPI_VERSION >= 3
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::scatter::call", environment);
    }
  };
}


# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_nothrow_copy_constructible
# undef YAMPI_is_same

#endif
