#ifndef YAMPI_SCATTER_HPP
# define YAMPI_SCATTER_HPP

# include <boost/config.hpp>

# ifdef BOOST_NO_CXX11_NULLPTR
#   include <cstddef>
# endif
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

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif


namespace yampi
{
  // TODO: implement MPI_Scatterv
  template <typename ContiguousIterator, typename ReceiveValue>
  inline void scatter(
    ContiguousIterator const first, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
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
          root.mpi_rank(), communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename SendValue, typename ReceiveValue>
  inline void scatter(
    ::yampi::buffer<SendValue> const send_buffer, ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator_base const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          receive_buffer.data(), receive_buffer.count(), receive_buffer.datatype().mpi_datatype(),
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename ReceiveValue>
  inline void scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::nonroot_call_on_root_error("yampi::scatter");

    ::yampi::scatter(nullptr, receive_buffer, root, communicator, environment);
  }

  template <typename ReceiveValue>
  inline void scatter(
    ::yampi::buffer<ReceiveValue> receive_buffer, ::yampi::rank const root,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  { ::yampi::scatter(nullptr, receive_buffer, root, communicator, environment); }

  template <typename Value>
  inline void scatter(
    ::yampi::in_place_t const,
    ::yampi::buffer<Value> const send_buffer, ::yampi::rank const root,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    if (communicator.rank(environment) == root)
      throw ::yampi::root_call_on_nonroot_error("yampi::scatter");

# if MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          const_cast<Value*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
          root.mpi_rank(), communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  template <typename SendValue>
  inline void scatter(
    ::yampi::buffer<SendValue> const send_buffer,
    ::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
# if MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          send_buffer.data(), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# else // MPI_VERSION >= 3
    int const error_code
      = MPI_Scatter(
          const_cast<SendValue*>(send_buffer.data()), send_buffer.count(), send_buffer.datatype().mpi_datatype(),
          nullptr, 0, MPI_DATATYPE_NULL,
          MPI_ROOT, communicator.mpi_comm());
# endif // MPI_VERSION >= 3
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }

  inline void scatter(::yampi::intercommunicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code
      = MPI_Scatter(
          nullptr, 0, MPI_DATATYPE_NULL, nullptr, 0, MPI_DATATYPE_NULL,
          MPI_PROC_NULL, communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::scatter", environment);
  }
}


# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif
# undef YAMPI_addressof
# undef YAMPI_is_same

#endif
