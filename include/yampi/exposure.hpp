#ifndef YAMPI_EXPOSURE_HPP
# define YAMPI_EXPOSURE_HPP

# include <utility>
# include <memory>

# include <mpi.h>

# include <yampi/window_base.hpp>
# include <yampi/group.hpp>
# include <yampi/assertion_mode.hpp>
# include <yampi/environment.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  class unexpected_exposure_status_error
    : public std::runtime_error
  {
   public:
    unexpected_exposure_status_error()
      : std::runtime_error("Error occurred because of strange exposure member variables")
    { }
  };


  struct defer_exposure_t { };
  struct adapt_exposure_t { };
# if __cplusplus >= 201703L
  inline constexpr ::yampi::defer_exposure_t defer_exposure{};
  inline constexpr ::yampi::adapt_exposure_t adapt_exposure{};
# else
  constexpr ::yampi::defer_exposure_t defer_exposure{};
  constexpr ::yampi::adapt_exposure_t adapt_exposure{};
# endif

  template <typename Window>
  class exposure_guard
  {
    ::yampi::window_base<Window>& window_;

   public:
    exposure_guard(exposure_guard const&) = delete;
    exposure_guard& operator=(exposure_guard const&) = delete;

    ~exposure_guard() noexcept { MPI_Win_wait(window_.mpi_win()); }

    exposure_guard(::yampi::group const& group, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_(window)
    { do_post(group, 0, environment); }

    exposure_guard(
      ::yampi::group const& group, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : window_(window)
    { do_post(group, static_assert<int>(assertion), environment); }

    exposure_guard(::yampi::window_base<Window>& window, ::yampi::adopt_exposure_t const)
      : window_(window)
    { }

   private:
    void do_post(::yampi::group const& group, int const assertion, ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Win_post(group.mpi_group(), assertion, window_.mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exposure_guard::do_post", environment);
    }
  };


  template <typename Window>
  class unique_exposure
  {
    ::yampi::window_base<Window>* window_ptr_;
    bool owns_;

   public:
    unique_exposure() noexcept : window_ptr_(nullptr), owns_(false) { }

    unique_exposure(unique_exposure const&) = delete;
    unique_exposure& operator=(unique_exposure const&) = delete;

    unique_exposure(unique_exposure&& other)
      : window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    unique_exposure& operator=(unique_exposure&& other)
    {
      if (this != std::addressof(other))
      {
        if (owns_)
          MPI_Win_wait(window_ptr_->mpi_win());
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }

    ~unique_exposure() noexcept
    {
      if (owns_)
        MPI_Win_wait(window_ptr_->mpi_win());
    }

    unique_exposure(
      ::yampi::group const& group, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_ptr_(std::addressof(window)), owns_(false)
    { post(group, environment); }

    unique_exposure(
      ::yampi::group const& group, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : window_ptr_(std::addressof(window)), owns_(false)
    { post(group, assertion, environment); }

    unique_exposure(::yampi::window_base<Window>& window, ::yampi::defer_exposure_t const)
      : window_ptr_(std::addressof(window)), owns_(false)
    { }

    unique_exposure(::yampi::window_base<Window>& window, ::yampi::adopt_exposure_t const)
      : window_ptr_(std::addressof(window)), owns_(true)
    { }

    void post(::yampi::group const& group, ::yampi::environment const& environment) const
    { do_post(group, 0, environment); }

    void post(::yampi::group const& group, ::yampi::assertion_mode const assertion, ::yampi::environment const& environment) const
    { do_post(group, static_cast<int>(assertion), environment); }

   private:
    void do_post(::yampi::group const& group, int const assertion, ::yampi::environment const& environment) const
    {
      if (owns_)
        throw ::yampi::unexpected_exposure_status_error(environment);

      int const error_code = MPI_Win_post(group.mpi_group(), assertion, window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exposure::do_post", environment);
      owns_ = true;
    }

   public:
    void wait(::yampi::environment const& environment) const
    {
      if (not owns_)
        throw ::yampi::unexpected_exposure_status_error(environment);

      int const error_code = MPI_Win_wait(window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exposure::wait", environment);
      owns_ = false;
    }

    bool test(::yampi::environment const& environment) const
    {
      if (not owns_)
        throw ::yampi::unexpected_exposure_status_error(environment);

      int result;
      int const error_code = MPI_Win_test(window_ptr_->mpi_win(), std::addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exposure::test", environment);

      if (static_cast<bool>(result))
        owns_ = false;

      return static_cast<bool>(result);
    }

    void swap(exposure& other) noexcept
    {
      using std::swap;
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::window_base<Window>* window_ptr() const noexcept { return window_ptr_; }
    bool owns_exposure() const noexcept { return owns_; }
  };

  template <typename Window>
  inline void swap(::yampi::exposure<Window>& lhs, ::yampi::exposure<Window>& rhs) noexcept
  { lhs.swap(rhs); }
}


#endif

