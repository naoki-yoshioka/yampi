#ifndef YAMPI_LOCK_HPP
# define YAMPI_LOCK_HPP

# include <utility>
# include <memory>

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/window_base.hpp>
# include <yampi/assertion_mode.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>


namespace yampi
{
  class unexpected_lock_status_error
    : public std::runtime_error
  {
   public:
    unexpected_lock_status_error()
      : std::runtime_error{"Error occurred because of strange lock member variables"}
    { }
  };


  struct defer_lock_t { };
  struct adapt_lock_t { };
# if __cplusplus >= 201703L
  inline constexpr ::yampi::defer_lock_t defer_lock{};
  inline constexpr ::yampi::adapt_lock_t adapt_lock{};
# else
  constexpr ::yampi::defer_lock_t defer_lock{};
  constexpr ::yampi::adapt_lock_t adapt_lock{};
# endif

  template <typename Window>
  class lock_guard
  {
    ::yampi::rank rank_;
    ::yampi::window_base<Window>& window_;

   public:
    lock_guard(lock_guard const&) = delete;
    lock_guard& operator=(lock_guard const&) = delete;

    ~lock_guard() noexcept { MPI_Win_unlock(rank_.mpi_rank(), window_.mpi_win()); }

    lock_guard(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : rank_{rank}, window_{window}
    { do_lock(0, environment); }

    lock_guard(
      ::yampi::rank const rank, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : rank_{rank}, window_{window}
    { do_lock(static_cast<int>(assertion), environment); }

    lock_guard(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::adopt_lock_t const)
      : rank_{rank}, window_{window}
    { }

   private:
    void do_lock(int const assertion, ::yampi::environment const& environment) const
    {
      int const error_code
        = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank_.mpi_rank(), assertion, window_.mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::lock_guard::do_lock", environment);
    }
  };


  template <typename Window>
  class unique_lock
  {
    ::yampi::rank rank_;
    ::yampi::window_base<Window>* window_ptr_;
    bool owns_;

   public:
    unique_lock() noexcept : rank_{}, window_ptr_{nullptr}, owns_{false} { }

    unique_lock(unique_lock const&) = delete;
    unique_lock& operator=(unique_lock const&) = delete;

    unique_lock(unique_lock&& other)
      : rank_{std::move(other.rank_)}, window_ptr_{std::move(other.window_ptr_)}, owns_{std::move(other.owns_)}
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    unique_lock& operator=(unique_lock&& other)
    {
      if (this != std::addressof(other))
      {
        if (owns_)
          MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
        rank_ = std::move(other.rank_);
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }

    ~unique_lock() noexcept
    {
      if (owns_)
        MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
    }

    explicit unique_lock(::yampi::rank const rank)
      : rank_{rank}, window_ptr_{nullptr}, owns_{false}
    { }

    unique_lock(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{false}
    { lock(environment); }

    unique_lock(
      ::yampi::rank const rank, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{false}
    { lock(assertion, environment); }

    unique_lock(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::defer_lock_t const)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{false}
    { }

    unique_lock(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::adopt_lock_t const)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{true}
    { }

    void lock(::yampi::environment const& environment) const
    { do_lock(0, environment); }

    void lock(::yampi::assertion_mode const assertion, ::yampi::environment const& environment) const
    { do_lock(static_cast<int>(assertion), environment); }

   private:
    void do_lock(int const assertion, ::yampi::environment const& environment) const
    {
      if (owns_)
        throw ::yampi::unexpected_lock_status_error(environment):

      int const error_code
        = MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank_.mpi_rank(), assertion, window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::unique_lock::do_lock", environment);
      owns_ = true;
    }

   public:
    void unlock(::yampi::environment const& environment) const
    {
      if (not owns_)
        throw ::yampi::unexpected_lock_status_error(environment);

      int const error_code
        = MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::unique_lock::unlock", environment);
      owns_ = false;
    }

    void swap(unique_lock& other) noexcept
    {
      using std::swap;
      swap(rank_, other.rank_);
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::rank const& rank() const noexcept { return rank_; }
    ::yampi::window_base<Window>* window_ptr() const noexcept { return window_ptr_; }
    bool owns_lock() const noexcept { return owns_; }
  };

  template <typename Window>
  inline void swap(::yampi::unique_lock<Window>& lhs, ::yampi::unique_lock<Window>& rhs) noexcept
  { lhs.swap(rhs); }


  template <typename Window>
  class shared_lock
  {
    ::yampi::rank rank_;
    ::yampi::window_base<Window>* window_ptr_;
    bool owns_;

   public:
    shared_lock() noexcept : rank_{}, window_ptr_{nullptr}, owns_{false} { }

    shared_lock(shared_lock const&) = delete;
    shared_lock& operator=(shared_lock const&) = delete;

    shared_lock(shared_lock&& other)
      : rank_{std::move(other.rank_)}, window_ptr_{std::move(other.window_ptr_)}, owns_{std::move(other.owns_)}
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    shared_lock& operator=(shared_lock&& other)
    {
      if (this != std::addressof(other))
      {
        rank_ = std::move(other.rank_);
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }

    ~shared_lock() noexcept
    {
      if (owns_)
        MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
    }

    explicit shared_lock(::yampi::rank const rank)
      : rank_{rank}, window_ptr_{nullptr}, owns_{false}
    { }

    shared_lock(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{false}
    { lock(environment); }

    shared_lock(
      ::yampi::rank const rank, ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{false}
    { lock(assertion, environment); }

    shared_lock(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::defer_lock_t const)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{false}
    { }

    shared_lock(::yampi::rank const rank, ::yampi::window_base<Window>& window, ::yampi::adopt_lock_t const)
      : rank_{rank}, window_ptr_{std::addressof(window)}, owns_{true}
    { }

    void lock(::yampi::environment const& environment) const
    { do_lock(0, environment); }

    void lock(::yampi::assertion_mode const assertion, ::yampi::environment const& environment) const
    { do_lock(static_cast<int>(assertion), environment); }

   private:
    void do_lock(int const assertion, ::yampi::environment const& environment) const
    {
      if (owns_)
        throw ::yampi::unexpected_lock_status_error(environment):

      int const error_code
        = MPI_Win_lock(MPI_LOCK_SHARED, rank_.mpi_rank(), assertion, window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::shared_lock::do_lock", environment);
      owns_ = true;
    }

   public:
    void unlock(::yampi::environment const& environment) const
    {
      if (not owns_)
        throw ::yampi::unexpected_lock_status_error(environment);

      int const error_code
        = MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::shared_lock::unlock", environment);
      owns_ = false;
    }

    void swap(shared_lock& other) noexcept
    {
      using std::swap;
      swap(rank_, other.rank_);
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::rank const& rank() const noexcept { return rank_; }
    ::yampi::window_base<Window>* window_ptr() const noexcept { return window_ptr_; }
    bool owns_lock() const noexcept { return owns_; }
  };

  template <typename Window>
  inline void swap(::yampi::shared_lock<Window>& lhs, ::yampi::shared_lock<Window>& rhs) noexcept
  { lhs.swap(rhs); }
# if MPI_VERSION >= 3


  template <typename Window>
  class all_shared_lock
  {
    ::yampi::window_base<Window>* window_ptr_;
    bool owns_;

   public:
    all_shared_lock() noexcept : window_ptr_{nullptr}, owns_{false} { }

    all_shared_lock(all_shared_lock const&) = delete;
    all_shared_lock& operator=(all_shared_lock const&) = delete;

    all_shared_lock(all_shared_lock&& other)
      : window_ptr_{std::move(other.window_ptr_)}, owns_{std::move(other.owns_)}
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    all_shared_lock& operator=(all_shared_lock&& other)
    {
      if (this != std::addressof(other))
      {
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }

    ~all_shared_lock() noexcept
    {
      if (owns_)
        MPI_Win_unlock_all(window_ptr_->mpi_win());
    }

    all_shared_lock(::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_ptr_{std::addressof(window)}, owns_{false}
    { lock(environment); }

    all_shared_lock(
      ::yampi::assertion_mode const assertion, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_ptr_{std::addressof(window)}, owns_{false}
    { lock(assertion, environment); }

    all_shared_lock(::yampi::window_base<Window>& window, ::yampi::defer_lock_t const)
      : window_ptr_{std::addressof(window)}, owns_{false}
    { }

    all_shared_lock(::yampi::window_base<Window>& window, ::yampi::adopt_lock_t const)
      : window_ptr_{std::addressof(window)}, owns_{true}
    { }

    void lock(::yampi::environment const& environment) const
    { do_lock(0, environment); }

    void lock(::yampi::assertion_mode const assertion, ::yampi::environment const& environment) const
    { do_lock(static_cast<int>(assertion), environment); }

   private:
    void do_lock(int const assertion, ::yampi::environment const& environment) const
    {
      if (owns_)
        throw ::yampi::unexpected_lock_status_error(environment):

      int const error_code = MPI_Win_lock_all(assertion, window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::all_shared_lock::do_lock", environment);
      owns_ = true;
    }

   public:
    void unlock(::yampi::environment const& environment) const
    {
      if (not owns_)
        throw ::yampi::unexpected_lock_status_error(environment);

      int const error_code = MPI_Win_unlock_all(window_ptr_->mpi_win());
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::all_shared_lock::unlock", environment);
      owns_ = false;
    }

    void swap(all_shared_lock& other) noexcept
    {
      using std::swap;
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::window_base<Window>* window_ptr() const noexcept { return window_ptr_; }
    bool owns_lock() const noexcept { return owns_; }
  };

  template <typename Window>
  inline void swap(::yampi::all_shared_lock<Window>& lhs, ::yampi::all_shared_lock<Window>& rhs) noexcept
  { lhs.swap(rhs); }
# endif // MPI_VERSION >= 3
}


#endif

