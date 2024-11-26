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

      auto const error_code = MPI_Request_free(std::addressof(mpi_request_));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::request_base::free", environment};
    }

    bool is_null() const
      noexcept(noexcept(mpi_request_ == MPI_REQUEST_NULL))
    { return mpi_request_ == MPI_REQUEST_NULL; }

    bool operator==(request_base const& other) const noexcept
    { return mpi_request_ == other.mpi_request_; }

    ::yampi::status wait(::yampi::environment const& environment)
    {
      MPI_Status mpi_status;
      auto const error_code = MPI_Wait(std::addressof(mpi_request_), std::addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? ::yampi::status(mpi_status)
        : throw ::yampi::error{error_code, "yampi::request_base::wait", environment};
    }

    void wait(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      auto const error_code = MPI_Wait(std::addressof(mpi_request_), MPI_STATUS_IGNORE);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::request_base::wait", environment};
    }

    boost::optional< ::yampi::status > test(::yampi::environment const& environment)
    {
      int flag;
      MPI_Status mpi_status;
      auto const error_code = MPI_Test(std::addressof(mpi_request_), std::addressof(flag), std::addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error{error_code, "yampi::request_base::test", environment};
    }

    bool test(::yampi::ignore_status_t const, ::yampi::environment const& environment)
    {
      int flag;
      auto const error_code = MPI_Test(std::addressof(mpi_request_), std::addressof(flag), MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error{error_code, "yampi::request_base::test", environment};
    }

    boost::optional< ::yampi::status > status(::yampi::environment const& environment) const
    {
      int flag;
      MPI_Status mpi_status;
      auto const error_code = MPI_Request_get_status(mpi_request_, std::addressof(flag), std::addressof(mpi_status));

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
          ? boost::make_optional(::yampi::status(mpi_status))
          : boost::none
        : throw ::yampi::error{error_code, "yampi::request_base::status", environment};
    }

    bool exists_status(::yampi::environment const& environment) const
    {
      int flag;
      auto const error_code = MPI_Request_get_status(mpi_request_, std::addressof(flag), MPI_STATUS_IGNORE);

      return error_code == MPI_SUCCESS
        ? static_cast<bool>(flag)
        : throw ::yampi::error{error_code, "yampi::request_base::exists_status", environment};
    }

    void cancel(::yampi::environment const& environment) const
    {
      auto const error_code = MPI_Cancel(const_cast<MPI_Request*>(std::addressof(mpi_request_)));

      if (error_code != MPI_SUCCESS)
        throw ::yampi::error{error_code, "yampi::request_base::cancel", environment};
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
}


# undef YAMPI_is_nothrow_swappable

#endif

