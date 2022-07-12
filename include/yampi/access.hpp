#ifndef YAMPI_ACCESS_HPP
# define YAMPI_ACCESS_HPP

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
  class unexpected_access_status_error
    : public std::runtime_error
  {
   public:
    unexpected_access_status_error()
      : std::runtime_error("Error occurred because of strange access member variables")
    { }
  };


  struct defer_access_t { };
  struct adapt_access_t { };
# if __cplusplus >= 201703L
  inline constexpr ::yampi::defer_access_t defer_access{};
  inline constexpr ::yampi::adapt_access_t adapt_access{};
# else
  constexpr ::yampi::defer_access_t defer_access{};
  constexpr ::yampi::adapt_access_t adapt_access{};
# endif

  template <typename Window>
  class access_guard
  {
    ::yampi::window_base<Window>& window_;

   public:
    access_guard(access_guard const&) = delete;
    access_guard& operator=(access_guard const&) = delete;

    ~access_guard() noexcept { MPI_Win_complete(window_.mpi_win()); }

    access_guard(::yampi::group const& group, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_(window)
    { do_start(group, 0, environment); }

    access_guard(
      ::yampi::group const& group, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : window_(window)
    { do_start(group, static_cast<int>(assertion), environment); }

    access_guard(::yampi::window_base<Window>& window, ::yampi::adopt_access_t const)
      : window_(window)
    { }

   private:
    void do_start(::yampi::group const& group, int const assertion, ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Win_start(group.mpi_group(), assertion, window_.mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::access_guard::do_start", environment);
    }
  };


  template <typename Window>
  class unique_access
  {
    ::yampi::window_base<Window>* window_ptr_;
    bool owns_;

   public:
    access() noexcept : window_ptr_(nullptr), owns_(false) { }

    unique_access(unique_access const&) = delete;
    unique_access& operator=(unique_access const&) = delete;

    unique_access(unique_access&& other)
      : window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    unique_access& operator=(unique_access&& other)
    {
      if (this != std::addressof(other))
      {
        if (owns_)
          MPI_Win_complete(window_ptr_->mpi_win());
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }

    ~unique_access() noexcept
    {
      if (owns_)
        MPI_Win_complete(window_ptr_->mpi_win());
    }

    unique_access(::yampi::group const& group, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_ptr_(std::addressof(window)), owns_(false)
    { start(group, environment); }

    unique_access(
      ::yampi::group const& group, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : window_ptr_(std::addressof(window)), owns_(false)
    { start(group, assertion, environment); }

    unique_access(::yampi::window_base<Window>& window, ::yampi::defer_access_t const)
      : window_ptr_(std::addressof(window)), owns_(false)
    { }

    unique_access(::yampi::window_base<Window>& window, ::yampi::adopt_access_t const)
      : window_ptr_(std::addressof(window)), owns_(true)
    { }

    void start(::yampi::group const& group, ::yampi::environment const& environment) const
    { do_start(group, 0, environment); }

    void start(::yampi::group const& group, ::yampi::assertion_mode const assertion, ::yampi::environment const& environment) const
    { do_start(group, static_cast<int>(assertion), environment); }

   private:
    void do_start(::yampi::group const& group, int const assertion, ::yampi::environment const& environment) const
    {
      if (owns_)
        throw ::yampi::unexpected_access_status_error(environment);

      int const error_code = MPI_Win_start(group.mpi_group(), assertion, window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::access::do_start", environment);
      owns_ = true;
    }

   public:
    void complete(::yampi::environment const& environment) const
    {
      if (not owns_)
        throw ::yampi::unexpected_access_status_error(environment);

      int const error_code = MPI_Win_complete(window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::access::complete", environment);
      owns_ = false;
    }

    void swap(access& other) noexcept
    {
      using std::swap;
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::window_base<Window>* window_ptr() const noexcept { return window_ptr_; }
    bool owns_access() const noexcept { return owns_; }
  };

  template <typename Window>
  inline void swap(::yampi::access<Window>& lhs, ::yampi::access<Window>& rhs) noexcept
  { lhs.swap(rhs); }
}


#endif

