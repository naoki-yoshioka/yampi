#ifndef YAMPI_WINDOW_HPP
# define YAMPI_WINDOW_HPP

# include <boost/config.hpp>

# ifdef BOOST_NO_CXX11_NULLPTR
#   include <cstddef>
# endif
# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/window.hpp>
# include <yampi/mode.hpp>
# include <yampi/rank.hpp>
# include <yampi/error.hpp>

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# ifdef BOOST_NO_CXX11_NULLPTR
#   define nullptr NULL
# endif

# ifndef BOOST_NO_CXX11_SCOPED_ENUMS
#   define YAMPI_MODE ::yampi::mode
# else // BOOST_NO_CXX11_SCOPED_ENUMS
#   define YAMPI_MODE ::yampi::mode::mode_
# endif // BOOST_NO_CXX11_SCOPED_ENUMS


namespace yampi
{
  class unexpected_lock_status_error
    : public std::runtime_error
  {
   public:
    unexpected_lock_status_error()
      : std::runtime_error("Error occurred because of strange lock member variables")
    { }
  };


  struct defer_lock_t { };
  struct adapt_lock_t { };

  class lock_guard
  {
    ::yampi::rank rank_;
    ::yampi::window& window_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    lock_guard(lock_guard const&) = delete;
    lock_guard& operator=(lock_guard const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    lock_guard(lock_guard const&);
    lock_guard& operator=(lock_guard const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

    ~lock_guard() BOOST_NOEXCEPT
    { MPI_Win_unlock(rank_.mpi_rank(), window_.mpi_win()); }

    lock_guard(::yampi::rank const& rank, ::yampi::window& window, ::yampi::environment const& environment)
      : rank_(rank), window_(window)
    { do_lock(0, environment); }

    lock_guard(
      ::yampi::rank const& rank, YAMPI_MODE const assertion, ::yampi::window& window,
      ::yampi::environment const& environment)
      : rank_(rank), window_(window)
    { do_lock(static_cast<int>(assertion), environment); }

    lock_guard(::yampi::rank const& rank, ::yampi::window& window, ::yampi::adopt_lock_t const)
      : rank_(rank), window_(window)
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


  class unique_lock
  {
    ::yampi::rank rank_;
    ::yampi::window* window_ptr_;
    bool owns_;

   public:
    unique_lock() BOOST_NOEXCEPT
      : rank_(), window_ptr_(nullptr), owns_(false)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    unique_lock(unique_lock const&) = delete;
    unique_lock& operator=(unique_lock const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    unique_lock(unique_lock const&);
    unique_lock& operator=(unique_lock const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    unique_lock(unique_lock&& other)
      : rank_(std::move(other.rank_)), window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    unique_lock& operator=(unique_lock&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        rank_ = std::move(other.rank_);
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~unique_lock() BOOST_NOEXCEPT
    {
      if (owns_)
        MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
    }

    explicit unique_lock(::yampi::rank const& rank)
      : rank_(rank), window_ptr_(nullptr), owns_(false)
    { }

    unique_lock(::yampi::rank const& rank, ::yampi::window& window, ::yampi::environment const& environment)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(false)
    { lock(environment); }

    unique_lock(
      ::yampi::rank const& rank, YAMPI_MODE const assertion, ::yampi::window& window,
      ::yampi::environment const& environment)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(false)
    { lock(assertion, environment); }

    unique_lock(::yampi::rank const& rank, ::yampi::window& window, ::yampi::defer_lock_t const)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(false)
    { }

    unique_lock(::yampi::rank const& rank, ::yampi::window& window, ::yampi::adopt_lock_t const)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(true)
    { }

    void lock(::yampi::environment const& environment) const
    { do_lock(0, environment); }

    void lock(YAMPI_MODE const assertion, ::yampi::environment const& environment) const
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

    void swap(unique_lock& other) BOOST_NOEXCEPT
    {
      using std::swap;
      swap(rank_, other.rank_);
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::rank const& rank() const BOOST_NOEXCEPT { return rank_; }
    ::yampi::window* window_ptr() const BOOST_NOEXCEPT { return window_ptr_; }
    bool owns_lock() const BOOST_NOEXCEPT { return owns_; }
  };

  inline void swap(::yampi::unique_lock& lhs, ::yampi::unique_lock& rhs) BOOST_NOEXCEPT
  { lhs.swap(rhs); }


  class shared_lock
  {
    ::yampi::rank rank_;
    ::yampi::window* window_ptr_;
    bool owns_;

   public:
    shared_lock() BOOST_NOEXCEPT
      : rank_(), window_ptr_(nullptr), owns_(false)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    shared_lock(shared_lock const&) = delete;
    shared_lock& operator=(shared_lock const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    shared_lock(shared_lock const&);
    shared_lock& operator=(shared_lock const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    shared_lock(shared_lock&& other)
      : rank_(std::move(other.rank_)), window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    shared_lock& operator=(shared_lock&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        rank_ = std::move(other.rank_);
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~shared_lock() BOOST_NOEXCEPT
    {
      if (owns_)
        MPI_Win_unlock(rank_.mpi_rank(), window_ptr_->mpi_win());
    }

    explicit shared_lock(::yampi::rank const& rank)
      : rank_(rank), window_ptr_(nullptr), owns_(false)
    { }

    shared_lock(::yampi::rank const& rank, ::yampi::window& window, ::yampi::environment const& environment)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(false)
    { lock(environment); }

    shared_lock(
      ::yampi::rank const& rank, YAMPI_MODE const assertion, ::yampi::window& window,
      ::yampi::environment const& environment)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(false)
    { lock(assertion, environment); }

    shared_lock(::yampi::rank const& rank, ::yampi::window& window, ::yampi::defer_lock_t const)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(false)
    { }

    shared_lock(::yampi::rank const& rank, ::yampi::window& window, ::yampi::adopt_lock_t const)
      : rank_(rank), window_ptr_(YAMPI_addressof(window)), owns_(true)
    { }

    void lock(::yampi::environment const& environment) const
    { do_lock(0, environment); }

    void lock(YAMPI_MODE const assertion, ::yampi::environment const& environment) const
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

    void swap(shared_lock& other) BOOST_NOEXCEPT
    {
      using std::swap;
      swap(rank_, other.rank_);
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::rank const& rank() const BOOST_NOEXCEPT { return rank_; }
    ::yampi::window* window_ptr() const BOOST_NOEXCEPT { return window_ptr_; }
    bool owns_lock() const BOOST_NOEXCEPT { return owns_; }
  };

  inline void swap(::yampi::shared_lock& lhs, ::yampi::shared_lock& rhs) BOOST_NOEXCEPT
  { lhs.swap(rhs); }


  class all_shared_lock
  {
    ::yampi::window* window_ptr_;
    bool owns_;

   public:
    all_shared_lock() BOOST_NOEXCEPT
      : window_ptr_(nullptr), owns_(false)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    all_shared_lock(all_shared_lock const&) = delete;
    all_shared_lock& operator=(all_shared_lock const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    all_shared_lock(all_shared_lock const&);
    all_shared_lock& operator=(all_shared_lock const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    all_shared_lock(all_shared_lock&& other)
      : window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    all_shared_lock& operator=(all_shared_lock&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        other.window_ptr_ = nullptr;
        other.owns_ = false;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~all_shared_lock() BOOST_NOEXCEPT
    {
      if (owns_)
        MPI_Win_unlock_all(window_ptr_->mpi_win());
    }

    all_shared_lock(::yampi::window& window, ::yampi::environment const& environment)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { lock(environment); }

    all_shared_lock(
      YAMPI_MODE const assertion, ::yampi::window& window, ::yampi::environment const& environment)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { lock(assertion, environment); }

    all_shared_lock(::yampi::window& window, ::yampi::defer_lock_t const)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { }

    all_shared_lock(::yampi::window& window, ::yampi::adopt_lock_t const)
      : window_ptr_(YAMPI_addressof(window)), owns_(true)
    { }

    void lock(::yampi::environment const& environment) const
    { do_lock(0, environment); }

    void lock(YAMPI_MODE const assertion, ::yampi::environment const& environment) const
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

    void swap(all_shared_lock& other) BOOST_NOEXCEPT
    {
      using std::swap;
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::window* window_ptr() const BOOST_NOEXCEPT { return window_ptr_; }
    bool owns_lock() const BOOST_NOEXCEPT { return owns_; }
  };

  inline void swap(::yampi::all_shared_lock& lhs, ::yampi::all_shared_lock& rhs) BOOST_NOEXCEPT
  { lhs.swap(rhs); }
}


# undef YAMPI_MODE
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef YAMPI_addressof

#endif

