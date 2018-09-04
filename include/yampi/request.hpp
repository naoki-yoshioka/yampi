#ifndef YAMPI_REQUEST_HPP
# define YAMPI_REQUEST_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class null_request_t { };

  class request
  {
    MPI_Request mpi_request_;

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    request() = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    request() : mpi_request_() { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    request(request const&) = delete;
    request& operator=(request const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    request(request const&);
    request& operator=(request const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    request(request&&) = default;
    request& operator=(request&&) = default;
#   else // BOOST_NO_CXX11_RVALUE_REFERENCES
    request(request&&) : mpi_request_(std::move(other.mpi_request_)) { }
    request& operator=(request&&)
    {
      if (this != YAMPI_addressof(other))
        mpi_request_ = std::move(other.mpi_request_);
      return *this;
    }
#   endif // BOOST_NO_CXX11_RVALUE_REFERENCES
# endif

    ~request() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_request_ == MPI_REQUEST_NULL or mpi_request_ == MPI_Request())
        return;

      MPI_Request_free(YAMPI_addressof(mpi_request_));
    }

    explicit request(MPI_Request const& req) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_request_(req)
    { }

    explicit request(::yampi::null_request_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_request_(MPI_REQUEST_NULL)
    { }

    void release(::yampi::environment const& environment)
    {
      if (mpi_request_ == MPI_REQUEST_NULL or mpi_request_ == MPI_Request())
        return;

      int const error_code = MPI_Request_free(YAMPI_addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::release", environment);
    }


    bool operator==(request const& other) const
    { return mpi_request_ == other.mpi_request_; }

    MPI_Request const& mpi_request() const { return mpi_request_; }
    void mpi_request(MPI_Request const& mpi_req) { mpi_request_ = mpi_req; }

    void swap(request& other)
      BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable<MPI_Request>::value ))
    {
      using std::swap;
      swap(mpi_request_, other.mpi_request_);
    }
  };

  inline bool operator!=(::yampi::request const& lhs, ::yampi::request const& rhs)
  { return not (lhs == rhs); }

  inline void swap(::yampi::request& lhs, ::yampi::request& rhs)
    BOOST_NOEXCEPT_IF(( ::yampi::utility::is_nothrow_swappable< ::yampi::request >::value ))
  { lhs.swap(rhs); }


  inline bool is_valid_request(::yampi::request const& self)
  { return self != ::yampi::request(null_request_t()); }
}


# undef YAMPI_addressof

#endif
