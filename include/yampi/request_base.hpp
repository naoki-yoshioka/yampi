#ifndef YAMPI_REQUEST_BASE_HPP
# define YAMPI_REQUEST_BASE_HPP

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

# include <yampi/status.hpp>
# include <yampi/environment.hpp>
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
  struct request_send_t { };
  struct request_receive_t { };

  inline BOOST_CONSTEXPR ::yampi::request_send_t request_send() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::request_send_t(); }

  inline BOOST_CONSTEXPR ::yampi::request_receive_t request_receive() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::request_receive_t(); }

  class request_base
  {
   protected:
    MPI_Request mpi_request_;

   public:
    request_base() BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Request>::value)
      : mpi_request_(MPI_REQUEST_NULL)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    request_base(request_base const&) = delete;
    request_base& operator=(request_base const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    request_base(request_base const&);
    request_base& operator=(request_base const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    request_base(request_base&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_constructible<MPI_Request>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Request>::value)
      : mpi_request_(std::move(other.mpi_request_))
    { other.mpi_request_ = MPI_REQUEST_NULL; }

    request_base& operator=(request_base&& other)
      BOOST_NOEXCEPT_IF(
        YAMPI_is_nothrow_move_assignable<MPI_Request>::value
        and YAMPI_is_nothrow_copy_assignable<MPI_Request>::value)
    {
      if (this != YAMPI_addressof(other))
      {
        if (mpi_request_ != MPI_REQUEST_NULL)
          MPI_Request_free(YAMPI_addressof(mpi_request_));
        mpi_request_ = std::move(other.mpi_request_);
        other.mpi_request_ = MPI_REQUEST_NULL;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

   protected:
    ~request_base() BOOST_NOEXCEPT_OR_NOTHROW
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      MPI_Request_free(YAMPI_addressof(mpi_request_));
    }

   public:
    explicit request_base(MPI_Request const& mpi_request)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Request>::value)
      : mpi_request_(mpi_request)
    { }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Request const& mpi_request, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_request_ = mpi_request;
    }

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    void reset(request_base&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_request_ = std::move(other.mpi_request_);
      other.mpi_request_ = MPI_REQUEST_NULL;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    void free(::yampi::environment const& environment)
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      int const error_code = MPI_Request_free(YAMPI_addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_base::free", environment);
    }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(mpi_request_ == MPI_REQUEST_NULL))
    { return mpi_request_ == MPI_REQUEST_NULL; }

    bool operator==(request_base const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_request_ == other.mpi_request_; }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      int const error_code
        = MPI_Wait(YAMPI_addressof(mpi_request_), YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? ::yampi::status(mpi_status)
        : throw ::yampi::error(error_code, "yampi::request_base::wait", environment);
    }

    void wait(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Wait(YAMPI_addressof(mpi_request_), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_base::wait", environment);
    }

    boost::optional< ::yampi::status > test(::yampi::environment const& environment)
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
        : throw ::yampi::error(error_code, "yampi::request_base::test", environment);
    }

    bool test(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int flag;
      int const error_code
        = MPI_Test(
            YAMPI_addressof(mpi_request_), &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request_base::test", environment);
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
        : throw ::yampi::error(error_code, "yampi::request_base::status", environment);
    }

    bool exists_status(::yampi::environment const& environment) const
    {
      int flag;
      int const error_code
        = MPI_Request_get_status(mpi_request_, &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request_base::exists_status", environment);
    }

    void cancel(::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Cancel(const_cast<MPI_Request*>(YAMPI_addressof(mpi_request_)));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_base::cancel", environment);
    }

    MPI_Request const& mpi_request() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_request_; }

    void swap(request_base& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Request>::value)
    {
      using std::swap;
      swap(mpi_request_, other.mpi_request_);
    }
  };

  inline bool operator!=(::yampi::request_base const& lhs, ::yampi::request_base const& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::request_base& lhs, ::yampi::request_base& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  class request_ref_base
  {
   protected:
    MPI_Request* mpi_request_ptr_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    request_ref_base() = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    request_ref_base();

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~request_ref_base() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~request_ref_base() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

    explicit request_ref_base(MPI_Request& mpi_request)
      : mpi_request_ptr_(YAMPI_addressof(mpi_request))
    { }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void free(::yampi::environment const& environment)
    {
      if (*mpi_request_ptr_ == MPI_REQUEST_NULL)
        return;

      int const error_code = MPI_Request_free(mpi_request_ptr_);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_ref_base::free", environment);
    }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(*mpi_request_ptr_ == MPI_REQUEST_NULL))
    { return *mpi_request_ptr_ == MPI_REQUEST_NULL; }

    bool operator==(request_ref_base const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_request_ptr_ == other.mpi_request_ptr_; }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      int const error_code = MPI_Wait(mpi_request_ptr_, YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? ::yampi::status(mpi_status)
        : throw ::yampi::error(error_code, "yampi::request_ref_base::wait", environment);
    }

    void wait(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int const error_code = MPI_Wait(mpi_request_ptr_, MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_ref_base::wait", environment);
    }

    boost::optional< ::yampi::status > test(::yampi::environment const& environment)
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code = MPI_Test(mpi_request_ptr_, &flag, YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error(error_code, "yampi::request_ref_base::test", environment);
    }

    bool test(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int flag;
      int const error_code = MPI_Test(mpi_request_ptr_, &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request_ref_base::test", environment);
    }

    boost::optional< ::yampi::status > status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Request_get_status(*mpi_request_ptr_, &flag, YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error(error_code, "yampi::request_ref_base::status", environment);
    }

    bool exists_status(::yampi::environment const& environment) const
    {
      int flag;
      int const error_code
        = MPI_Request_get_status(*mpi_request_ptr_, &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request_ref_base::exists_status", environment);
    }

    void cancel(::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Cancel(const_cast<MPI_Request*>(mpi_request_ptr_));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_ref_base::cancel", environment);
    }

    MPI_Request const& mpi_request() const BOOST_NOEXCEPT_OR_NOTHROW { return *mpi_request_ptr_; }

    void swap(request_ref_base& other) BOOST_NOEXCEPT_OR_NOTHROW
    {
      using std::swap;
      swap(mpi_request_ptr_, other.mpi_request_ptr_);
    }
  };

  inline bool operator!=(::yampi::request_ref_base const& lhs, ::yampi::request_ref_base const& rhs) BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline void swap(::yampi::request_ref_base& lhs, ::yampi::request_ref_base& rhs) BOOST_NOEXCEPT_OR_NOTHROW
  { lhs.swap(rhs); }

  class request_cref_base
  {
   protected:
    MPI_Request const* mpi_request_ptr_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    request_cref_base() = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    request_cref_base();

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~request_cref_base() BOOST_NOEXCEPT_OR_NOTHROW = default;
# else // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    ~request_cref_base() BOOST_NOEXCEPT_OR_NOTHROW { }
# endif // BOOST_NO_CXX11_DEFAULTED_FUNCTIONS

    explicit request_cref_base(MPI_Request const& mpi_request)
      : mpi_request_ptr_(YAMPI_addressof(mpi_request))
    { }

    explicit request_cref_base(request_ref_base const& request)
      : mpi_request_ptr_(YAMPI_addressof(request.mpi_request()))
    { }

    bool is_null() const
      BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(*mpi_request_ptr_ == MPI_REQUEST_NULL))
    { return *mpi_request_ptr_ == MPI_REQUEST_NULL; }

    bool operator==(request_cref_base const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_request_ptr_ == other.mpi_request_ptr_; }

    boost::optional< ::yampi::status > status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Request_get_status(*mpi_request_ptr_, &flag, YAMPI_addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error(error_code, "yampi::request_cref_base::status", environment);
    }

    bool exists_status(::yampi::environment const& environment) const
    {
      int flag;
      int const error_code
        = MPI_Request_get_status(*mpi_request_ptr_, &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request_cref_base::exists_status", environment);
    }

    void cancel(::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Cancel(const_cast<MPI_Request*>(mpi_request_ptr_));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_cref_base::cancel", environment);
    }

    MPI_Request const& mpi_request() const BOOST_NOEXCEPT_OR_NOTHROW { return *mpi_request_ptr_; }
  };

  inline bool operator!=(::yampi::request_cref_base const& lhs, ::yampi::request_cref_base const& rhs) BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_nothrow_move_assignable
# undef YAMPI_is_nothrow_move_constructible
# undef YAMPI_is_nothrow_copy_assignable
# undef YAMPI_is_nothrow_copy_constructible

#endif

