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

# include <yampi/buffer.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# if MPI_VERSION >= 3
#   include <yampi/request_base.hpp>
# endif
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

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
    void call(::yampi::buffer<Value> buffer, ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Bcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root_.mpi_rank(), communicator_.mpi_comm());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast::call", environment);
    }
  };
# if MPI_VERSION >= 3

  class broadcast_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    broadcast_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    broadcast_request(broadcast_request const&) = delete;
    broadcast_request& operator=(broadcast_request const&) = delete;
#   else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    broadcast_request(broadcast_request const&);
    broadcast_request& operator=(broadcast_request const&);

   public:
#   endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    broadcast_request(broadcast_request&&) = default;
    broadcast_request& operator=(broadcast_request&&) = default;
#     endif
    ~broadcast_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    template <typename Value>
    broadcast_request(
      ::yampi::buffer<Value> buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_broadcast_request(buffer, root, communicator, environment))
    { }

   private:
    template <typename Value>
    static void do_broadcast(
      MPI_Request& mpi_request,
      ::yampi::buffer<Value> buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ibcast(
            buffer.data(), buffer.count(), buffer.datatype().mpi_datatype(),
            root.mpi_rank(), communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::broadcast_request::do_broadcast", environment);
    }

    template <typename Value>
    static MPI_Request make_broadcast_request(
      ::yampi::buffer<Value> buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_broadcast(result, buffer, root, communicator, environment);
      return result;
    }

   public:
    template <typename Value>
    void reset(
      ::yampi::buffer<Value> buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      broadcast(buffer, root, communicator, environment);
    }

    template <typename Value>
    void broadcast(
      ::yampi::buffer<Value> buffer, ::yampi::rank const& root,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_broadcast(mpi_request_, buffer, root, communicator, environment); }
  };
# endif // MPI_VERSION >= 3
}


# if MPI_VERSION >= 3
#   undef YAMPI_addressof
# endif
# undef YAMPI_is_nothrow_copy_constructible

#endif

