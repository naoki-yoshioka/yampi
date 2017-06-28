#ifndef YAMPI_REQUEST_HPP
# define YAMPI_REQUEST_HPP

# include <boost/config.hpp>

# include <mpi.h>


namespace yampi
{
  class null_request_t { };

  class request
  {
    MPI_Request req_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    request() = delete;
# else
   private:
    request();

   public:
# endif

    explicit request(MPI_Request const req) BOOST_NOEXCEPT_OR_NOTHROW
      : req_(req)
    { }

    explicit request(::yampi::null_request_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : req_(MPI_REQUEST_NULL)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    request(request const&) = default;
    request& operator=(request const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    request(request&&) = default;
    request& operator=(request&&) = default;
#   endif
    ~request() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    MPI_Request const& mpi_request() const { return req_; }
  };
}


#endif
