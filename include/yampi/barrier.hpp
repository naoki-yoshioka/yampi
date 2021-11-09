#ifndef YAMPI_BARRIER_HPP
# define YAMPI_BARRIER_HPP

# include <boost/config.hpp>

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#   else
#     include <boost/type_traits/has_nothrow_copy.hpp>
#   endif
#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     include <memory>
#   else
#     include <boost/core/addressof.hpp>
#   endif
# endif

# include <mpi.h>

# if MPI_VERSION >= 3
#   include <yampi/request_base.hpp>
# endif
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if MPI_VERSION >= 3
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   else
#     define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   endif

#   ifndef BOOST_NO_CXX11_ADDRESSOF
#     define YAMPI_addressof std::addressof
#   else
#     define YAMPI_addressof boost::addressof
#   endif
# endif


namespace yampi
{
  inline void barrier(::yampi::communicator const& communicator, ::yampi::environment const& environment)
  {
    int const error_code = MPI_Barrier(communicator.mpi_comm());
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::barrier", environment);
  }
# if MPI_VERSION >= 3

  class barrier_request
    : public ::yampi::request_base
  {
    typedef request_base base_type;

   public:
    barrier_request() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<base_type>::value)
      : base_type()
    { }

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    barrier_request(barrier_request const&) = default;
    barrier_request& operator=(barrier_request const&) = default;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    barrier_request(barrier_request&&) = default;
    barrier_request& operator=(barrier_request&&) = default;
#     endif
    ~barrier_request() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    barrier_request(::yampi::communicator const& communicator, ::yampi::environment const& environment)
      : base_type(make_barrier_request(communicator, environment))
    { }

   private:
    static void do_barrier(
      MPI_Request& mpi_request,
      ::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Ibarrier(communicator.mpi_comm(), YAMPI_addressof(mpi_request));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::barrier_request::do_barrier", environment);
    }

    static MPI_Request make_barrier_request(::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      MPI_Request result;
      do_barrier(result, communicator, environment);
      return result;
    }

   public:
    void reset(::yampi::communicator const& communicator, ::yampi::environment const& environment)
    {
      free(environment);
      barrier(communicator, environment);
    }

    void barrier(::yampi::communicator const& communicator, ::yampi::environment const& environment)
    { do_barrier(mpi_request_, communicator, environment); }
  };
# endif // MPI_VERSION >= 3
}


# if MPI_VERSION >= 3
#   undef YAMPI_addressof
#   undef YAMPI_is_nothrow_copy_constructible
# endif

#endif

