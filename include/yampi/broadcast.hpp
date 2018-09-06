#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

# include <boost/config.hpp>

# include <mpi.h>

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <yampi/environment.hpp>
# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
# endif

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define YAMPI_addressof std::addressof
#   else
#     define YAMPI_addressof boost::addressof
#   endif
# endif


namespace yampi
{
  class broadcast
  {
    ::yampi::communicator const& communicator_;
    ::yampi::rank root_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    broadcast() = delete;
    broadcast(broadcast const&) = delete;
    broadcast& operator=(broadcast const&) = delete;
# else
   private:
    broadcast();
    broadcast(broadcast const&);
    broadcast& operator=(broadcast const&);

   public:
# endif

    broadcast(
      ::yampi::communicator const& communicator, ::yampi::rank const root)
      BOOST_NOEXCEPT_OR_NOTHROW
      : communicator_(communicator), root_(root)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    broadcast(broadcast&&) = default;
    broadcast& operator=(broadcast&&) = default;
#   endif
    ~broadcast() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif


    template <typename Value>
    void call(::yampi::environment const& environment, ::yampi::buffer<Value>& buffer) const
    {
      int const error_code
        = MPI_Bcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);
    }

    template <typename Value>
    void call(::yampi::environment const& environment, ::yampi::buffer<Value> const& buffer) const
    {
      int const error_code
        = MPI_Bcast(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);
    }
# if MPI_VERSION >= 3

    template <typename Value>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<Value>& buffer, ::yampi::request& request) const
    {
      MPI_Request mpi_request;
      int const error_code
        = MPI_Ibcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);

      request.release(environment);
      request.mpi_request(mpi_request);
    }

    template <typename Value>
    void call(
      ::yampi::environment const& environment,
      ::yampi::buffer<Value> const& buffer, ::yampi::request& request) const
    {
      MPI_Request mpi_request;
      int const error_code
        = MPI_Ibcast(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);

      request.release(environment);
      request.mpi_request(mpi_request);
    }
# endif
  };
}


# if MPI_VERSION >= 3
#   undef YAMPI_addressof
# endif

#endif

