#ifndef YAMPI_BROADCAST_HPP
# define YAMPI_BROADCAST_HPP

# include <boost/config.hpp>

# include <mpi.h>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
# endif
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
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/error.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
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
    ::yampi::rank root_;
    ::yampi::communicator const& communicator_;

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

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    broadcast(broadcast&&) = default;
    broadcast& operator=(broadcast&&) = default;
#   endif
    ~broadcast() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    broadcast(
      ::yampi::rank const& root, ::yampi::communicator const& communicator)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value)
      : root_(root), communicator_(communicator)
    { }


    template <typename Value>
    void call(::yampi::buffer<Value>& buffer, ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Bcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);
    }

    template <typename Value>
    void call(::yampi::buffer<Value> const& buffer, ::yampi::environment const& environment) const
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
      ::yampi::request& request, ::yampi::buffer<Value>& buffer,
      ::yampi::environment const& environment) const
    {
      MPI_Request mpi_request;
      int const error_code
        = MPI_Ibcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);

      request.reset(mpi_request, environment);
    }

    template <typename Value>
    void call(
      ::yampi::request& request, ::yampi::buffer<Value> const& buffer,
      ::yampi::environment const& environment) const
    {
      MPI_Request mpi_request;
      int const error_code
        = MPI_Ibcast(
            const_cast<Value*>(buffer.data()), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);

      request.reset(mpi_request, environment);
    }
# endif
  };
}


# if MPI_VERSION >= 3
#   undef YAMPI_addressof
# endif
# undef YAMPI_is_nothrow_copy_constructible

#endif

