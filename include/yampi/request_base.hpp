#ifndef YAMPI_REQUEST_BASE_HPP
# define YAMPI_REQUEST_BASE_HPP

# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <memory>

# include <mpi.h>

# include <yampi/status.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct request_send_t { };
  struct request_receive_t { };
# if __cplusplus >= 201703L
  inline constexpr ::yampi::request_send_t request_send{};
  inline constexpr ::yampi::request_receive_t request_receive{};
# else
  constexpr ::yampi::request_send_t request_send{};
  constexpr ::yampi::request_receive_t request_receive{};
# endif

  class request_base
  {
   protected:
    MPI_Request mpi_request_;

   public:
    request_base() noexcept(std::is_nothrow_copy_constructible<MPI_Request>::value)
      : mpi_request_{MPI_REQUEST_NULL}
    { }

    request_base(request_base const&) = delete;
    request_base& operator=(request_base const&) = delete;

    request_base(request_base&& other)
      noexcept(
        std::is_nothrow_move_constructible<MPI_Request>::value
        and std::is_nothrow_copy_assignable<MPI_Request>::value)
      : mpi_request_{std::move(other.mpi_request_)}
    { other.mpi_request_ = MPI_REQUEST_NULL; }

    request_base& operator=(request_base&& other)
      noexcept(
        std::is_nothrow_move_assignable<MPI_Request>::value
        and std::is_nothrow_copy_assignable<MPI_Request>::value)
    {
      if (this != std::addressof(other))
      {
        if (mpi_request_ != MPI_REQUEST_NULL)
          MPI_Request_free(std::addressof(mpi_request_));
        mpi_request_ = std::move(other.mpi_request_);
        other.mpi_request_ = MPI_REQUEST_NULL;
      }
      return *this;
    }

   protected:
    ~request_base() noexcept
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      MPI_Request_free(std::addressof(mpi_request_));
    }

   public:
    explicit request_base(MPI_Request const& mpi_request)
      noexcept(std::is_nothrow_copy_constructible<MPI_Request>::value)
      : mpi_request_{mpi_request}
    { }

    void reset(::yampi::environment const& environment)
    { free(environment); }

    void reset(MPI_Request const& mpi_request, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_request_ = mpi_request;
    }

    void reset(request_base&& other, ::yampi::environment const& environment)
    {
      free(environment);
      mpi_request_ = std::move(other.mpi_request_);
      other.mpi_request_ = MPI_REQUEST_NULL;
    }

    void free(::yampi::environment const& environment)
    {
      if (mpi_request_ == MPI_REQUEST_NULL)
        return;

      int const error_code = MPI_Request_free(std::addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_base::free", environment);
    }

    bool is_null() const
      noexcept(noexcept(mpi_request_ == MPI_REQUEST_NULL))
    { return mpi_request_ == MPI_REQUEST_NULL; }

    bool operator==(request_base const& other) const noexcept
    { return mpi_request_ == other.mpi_request_; }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      int const error_code
        = MPI_Wait(std::addressof(mpi_request_), std::addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? ::yampi::status(mpi_status)
        : throw ::yampi::error(error_code, "yampi::request_base::wait", environment);
    }

    void wait(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int const error_code
        = MPI_Wait(std::addressof(mpi_request_), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_base::wait", environment);
    }

    boost::optional< ::yampi::status > test(::yampi::environment const& environment)
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Test(
            std::addressof(mpi_request_), &flag, std::addressof(mpi_status));

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
            std::addressof(mpi_request_), &flag, MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error(error_code, "yampi::request_base::test", environment);
    }

    boost::optional< ::yampi::status > status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Request_get_status(mpi_request_, &flag, std::addressof(mpi_status));

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

    [[deprecated]]
    void cancel(::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Cancel(const_cast<MPI_Request*>(std::addressof(mpi_request_)));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_base::cancel", environment);
    }

    MPI_Request const& mpi_request() const noexcept { return mpi_request_; }

    void swap(request_base& other)
      noexcept(YAMPI_is_nothrow_swappable<MPI_Request>::value)
    {
      using std::swap;
      swap(mpi_request_, other.mpi_request_);
    }
  };

  inline bool operator!=(::yampi::request_base const& lhs, ::yampi::request_base const& rhs) noexcept(noexcept(lhs == rhs))
  { return not (lhs == rhs); }

  inline void swap(::yampi::request_base& lhs, ::yampi::request_base& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }

  class request_ref_base
  {
   protected:
    MPI_Request* mpi_request_ptr_;

   public:
    request_ref_base() = delete;
    ~request_ref_base() noexcept = default;

    explicit request_ref_base(MPI_Request& mpi_request)
      : mpi_request_ptr_{std::addressof(mpi_request)}
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

    bool is_null() const noexcept(noexcept(*mpi_request_ptr_ == MPI_REQUEST_NULL))
    { return *mpi_request_ptr_ == MPI_REQUEST_NULL; }

    bool operator==(request_ref_base const& other) const noexcept { return mpi_request_ptr_ == other.mpi_request_ptr_; }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      int const error_code = MPI_Wait(mpi_request_ptr_, std::addressof(mpi_status));

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
      int const error_code = MPI_Test(mpi_request_ptr_, &flag, std::addressof(mpi_status));

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
        = MPI_Request_get_status(*mpi_request_ptr_, &flag, std::addressof(mpi_status));

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

    [[deprecated]]
    void cancel(::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Cancel(const_cast<MPI_Request*>(mpi_request_ptr_));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_ref_base::cancel", environment);
    }

    MPI_Request const& mpi_request() const noexcept { return *mpi_request_ptr_; }

    void swap(request_ref_base& other) noexcept
    {
      using std::swap;
      swap(mpi_request_ptr_, other.mpi_request_ptr_);
    }
  };

  inline bool operator!=(::yampi::request_ref_base const& lhs, ::yampi::request_ref_base const& rhs) noexcept
  { return not (lhs == rhs); }

  inline void swap(::yampi::request_ref_base& lhs, ::yampi::request_ref_base& rhs) noexcept
  { lhs.swap(rhs); }

  class request_cref_base
  {
   protected:
    MPI_Request const* mpi_request_ptr_;

   public:
    request_cref_base() = delete;
    ~request_cref_base() noexcept = default;

    explicit request_cref_base(MPI_Request const& mpi_request)
      : mpi_request_ptr_{std::addressof(mpi_request)}
    { }

    explicit request_cref_base(request_ref_base const& request)
      : mpi_request_ptr_{std::addressof(request.mpi_request())}
    { }

    bool is_null() const noexcept(noexcept(*mpi_request_ptr_ == MPI_REQUEST_NULL))
    { return *mpi_request_ptr_ == MPI_REQUEST_NULL; }

    bool operator==(request_cref_base const& other) const noexcept
    { return mpi_request_ptr_ == other.mpi_request_ptr_; }

    boost::optional< ::yampi::status > status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      int const error_code
        = MPI_Request_get_status(*mpi_request_ptr_, &flag, std::addressof(mpi_status));

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

    [[deprecated]]
    void cancel(::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Cancel(const_cast<MPI_Request*>(mpi_request_ptr_));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::request_cref_base::cancel", environment);
    }

    MPI_Request const& mpi_request() const noexcept { return *mpi_request_ptr_; }
  };

  inline bool operator!=(::yampi::request_cref_base const& lhs, ::yampi::request_cref_base const& rhs) noexcept
  { return not (lhs == rhs); }
}


# undef YAMPI_is_nothrow_swappable

#endif

