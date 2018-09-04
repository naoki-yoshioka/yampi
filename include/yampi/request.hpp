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

# include <yampi/status.hpp>
# include <yampi/error.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif


namespace yampi
{
  class request
  {
    MPI_Request mpi_request_;

   public:
    request() : mpi_request_(MPI_REQUEST_NULL) { }
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
    request(request&& other)
      : mpi_request_(std::move(other.mpi_request_))
    { other.mpi_request_ = MPI_REQUEST_NULL; }

    request& operator=(request&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        mpi_request_ = std::move(other.mpi_request_);
        other.mpi_request_ = MPI_REQUEST_NULL;
      }
      return *this;
    }
# endif

    ~request() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      MPI_Request_free(YAMPI_addressof(mpi_request_));
    }

    explicit request(MPI_Request const& req) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_request_(req)
    { }

    void release(::yampi::environment const& environment)
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      int const error_code = MPI_Request_free(YAMPI_addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::release", environment);
    }


    bool is_null() const { return mpi_request_ == MPI_REQUEST_NULL; }

    bool operator==(request const& other) const
    { return mpi_request_ == other.mpi_request_; }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      int const error_code
        = MPI_Wait(YAMPI_addressof(mpi_request_), YAMPI_addressof(mpi_status));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::wait", environment);

      return ::yampi::status(mpi_status);
    }

    void wait(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Wait(YAMPI_addressof(mpi_request_), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::wait", environment);
    }

    std::pair<bool, ::yampi::status> test(::yampi::environment const& environment)
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Test(
            YAMPI_addressof(mpi_request_), &flag, YAMPI_addressof(mpi_status));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::test", environment);

      return std::make_pair(static_cast<bool>(flag), ::yampi::status(mpi_status));
    }

    bool test(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int flag;
      int const error_code
        = MPI_Test(
            YAMPI_addressof(mpi_request_), &flag, MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::test", environment);

      return static_cast<bool>(flag);
    }

    std::pair<bool, ::yampi::status> status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Request_get_status(mpi_request_, &flag, YAMPI_addressof(mpi_status));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::status", environment);

      return std::make_pair(static_cast<bool>(flag), ::yampi::status(mpi_status));
    }


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
}


# undef YAMPI_addressof

#endif
