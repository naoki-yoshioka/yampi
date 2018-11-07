#ifndef YAMPI_INITIALIZE_RECEIVE_HPP
# define YAMPI_INITIALIZE_RECEIVE_HPP

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
  inline void initialize_receive(
    ::yampi::request& request,
    ::yampi::buffer<Value>& buffer, ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Recv_init(
          buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::initialize_receive", environment);

    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void initialize_receive(
    ::yampi::request& request,
    ::yampi::buffer<Value>& buffer, ::yampi::rank const source,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { ::yampi::initialize_receive(request, buffer, source, ::yampi::any_tag(), communicator, environment); }

  template <typename Value>
  inline void initialize_receive(
    ::yampi::request& request, ::yampi::buffer<Value>& buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { ::yampi::initialize_receive(request, buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }

  template <typename Value>
  inline void initialize_receive(
    ::yampi::request& request,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const source, ::yampi::tag const tag,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    MPI_Request mpi_request;
    int const error_code
      = MPI_Recv_init(
          const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
          source.mpi_rank(), tag.mpi_tag(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::initialize_receive", environment);

    request.reset(mpi_request, environment);
  }

  template <typename Value>
  inline void initialize_receive(
    ::yampi::request& request,
    ::yampi::buffer<Value> const& buffer, ::yampi::rank const source,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { ::yampi::initialize_receive(request, buffer, source, ::yampi::any_tag(), communicator, environment); }

  template <typename Value>
  inline void initialize_receive(
    ::yampi::request& request, ::yampi::buffer<Value> const& buffer,
    ::yampi::communicator const& communicator, ::yampi::environment const& environment)
  { ::yampi::initialize_receive(request, buffer, ::yampi::any_source(), ::yampi::any_tag(), communicator, environment); }
}


# undef YAMPI_addressof

#endif

