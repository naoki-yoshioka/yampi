#ifndef YAMPI_NONBLOCKING_RECEIVE_HPP
# define YAMPI_NONBLOCKING_RECEIVE_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/request.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  template <typename Value>
  inline void nonblocking_receive(
    ::yampi::communicator const& communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value>& buffer,
    ::yampi::request& request,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const tag = ::yampi::any_tag())
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Irecv(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_receive", environment);

    request.mpi_request(mpi_request);
  }

  template <typename Value>
  inline void nonblocking_receive(
    ::yampi::communicator const& communicator, ::yampi::environment const& environment,
    ::yampi::buffer<Value> const& buffer,
    ::yampi::request& request,
    ::yampi::rank const source = ::yampi::any_source(),
    ::yampi::tag const tag = ::yampi::any_tag())
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Irecv(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(),
          YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::nonblocking_receive", environment);

    request.mpi_request(mpi_request);
  }
}


# undef YAMPI_addressof

#endif

