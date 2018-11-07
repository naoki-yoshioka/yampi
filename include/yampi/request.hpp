#ifndef YAMPI_REQUEST_HPP
# define YAMPI_REQUEST_HPP

# include <boost/config.hpp>

# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/type_traits/has_nothrow_assign.hpp>
#   include <boost/type_traits/is_nothrow_move_constructible.hpp>
#   include <boost/type_traits/is_nothrow_move_assignable.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <boost/optional.hpp>

# include <yampi/status.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_is_nothrow_copy_assignable std::is_nothrow_copy_assignable
#   define YAMPI_is_nothrow_move_constructible std::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable std::is_nothrow_move_assignable
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_is_nothrow_copy_assignable boost::has_nothrow_assign
#   define YAMPI_is_nothrow_move_constructible boost::is_nothrow_move_constructible
#   define YAMPI_is_nothrow_move_assignable boost::is_nothrow_move_assignable
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif

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
    request()
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Request>::value)
      : mpi_request_(MPI_REQUEST_NULL)
    { }

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
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Request>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Request>::value)
      : mpi_request_(std::move(other.mpi_request_))
    { other.mpi_request_ = MPI_REQUEST_NULL; }

    request& operator=(request&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Request>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Request>::value)
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

    explicit request(MPI_Request const& req)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Request>::value)
      : mpi_request_(req)
    { }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Request const& req, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_request_ = req;
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      int const error_code = MPI_Request_free(YAMPI_addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::free", environment);
    }


    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_request_ == MPI_REQUEST_NULL))
    { return mpi_request_ == MPI_REQUEST_NULL; }

    bool operator==(request const& other) const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_request_ == other.mpi_request_))
    { return mpi_request_ == other.mpi_request_; }

    void start(::yampi::environment const& environment)
    {
      int const error_code = MPI_Start(YAMPI_addressof(mpi_request_));
      if (error_code == MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::start", environment);
    }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      int const error_code
        = MPI_Wait(YAMPI_addressof(mpi_request_), YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? ::yampi::status(mpi_status)
        : throw ::yampi::error(error_code, "yampi::request::wait", environment);
    }

    void wait(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Wait(YAMPI_addressof(mpi_request_), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request::wait", environment);
    }

    boost::optional< ::yampi::status >  test(::yampi::environment const& environment)
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Test(
            YAMPI_addressof(mpi_request_), &flag, YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error(error_code, "yampi::request::test", environment);

    }

    bool test(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int flag;
      int const error_code
        = MPI_Test(
            YAMPI_addressof(mpi_request_), &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request::test", environment);
    }

    boost::optional< ::yampi::status > status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Request_get_status(mpi_request_, &flag, YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error(error_code, "yampi::request::status", environment);
    }

    bool exists_status(::yampi::environment const& environment) const
    {
      int flag;
      int const error_code
        = MPI_Request_get_status(mpi_request_, &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request::status", environment);
    }


    MPI_Request const& mpi_request() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_request_; }
    void mpi_request(MPI_Request const& mpi_req)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_assignable<MPI_Request>::value)
    { mpi_request_ = mpi_req; }

    void swap(request& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Request>::value)
    {
      using std::swap;
      swap(mpi_request_, other.mpi_request_);
    }
  };

  inline bool operator!=(::yampi::request const& lhs, ::yampi::request const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::request& lhs, ::yampi::request& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif
