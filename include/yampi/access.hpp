#ifndef YAMPI_ACCESS_HPP
# define YAMPI_ACCESS_HPP

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

# include <yampi/window.hpp>
# include <yampi/group.hpp>
# include <yampi/mode.hpp>
# include <yampi/environment.hpp>
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

  class access_guard
  {
    ::yampi::window& window_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    access_guard(access_guard const&) = delete;
    access_guard& operator=(access_guard const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    access_guard(access_guard const&);
    access_guard& operator=(access_guard const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

    ~access_guard() BOOST_NOEXCEPT
    { MPI_Win_complete(window_.mpi_win()); }

    access_guard(::yampi::group const& group, ::yampi::window& window, ::yampi::environment const& environment)
      : window_(window)
    { do_start(group, 0, environment); }

    access_guard(
      ::yampi::group const& group, YAMPI_MODE const assertion, ::yampi::window& window,
      ::yampi::environment const& environment)
      : window_(window)
    { do_start(group, static_cast<int>(assertion), environment); }

    access_guard(::yampi::window& window, ::yampi::adopt_access_t const)
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


  class unique_access
  {
    ::yampi::window* window_ptr_;
    bool owns_;

   public:
    access() BOOST_NOEXCEPT
      : window_ptr_(nullptr), owns_(false)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    unique_access(unique_access const&) = delete;
    unique_access& operator=(unique_access const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    unique_access(unique_access const&);
    unique_access& operator=(unique_access const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    unique_access(unique_access&& other)
      : window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    unique_access& operator=(unique_access&& other)
    {
      if (this != YAMPI_addressof(other))
      {
        window_ptr_ = std::move(other.window_ptr_);
        owns_ = std::move(other.owns_);
        window_ptr_ = nullptr;
        owns_ = false;
      }
      return *this;
    }
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~unique_access() BOOST_NOEXCEPT
    {
      if (owns_)
        MPI_Win_complete(window_ptr_->mpi_win());
    }

    unique_access(::yampi::group const& group, ::yampi::window& window, ::yampi::environment const& environment)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { start(group, environment); }

    unique_access(
      ::yampi::group const& group, YAMPI_MODE const assertion, ::yampi::window& window,
      ::yampi::environment const& environment)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { start(group, assertion, environment); }

    unique_access(::yampi::window& window, ::yampi::defer_access_t const)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { }

    unique_access(::yampi::window& window, ::yampi::adopt_access_t const)
      : window_ptr_(YAMPI_addressof(window)), owns_(true)
    { }

    void start(::yampi::group const& group, ::yampi::environment const& environment) const
    { do_start(group, 0, environment); }

    void start(::yampi::group const& group, YAMPI_MODE const assertion, ::yampi::environment const& environment) const
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

    void swap(access& other) BOOST_NOEXCEPT
    {
      using std::swap;
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::window* window_ptr() const BOOST_NOEXCEPT_OR_NOTHROW { return window_ptr_; }
    bool owns_access() const BOOST_NOEXCEPT_OR_NOTHROW { return owns_; }
  };

  inline void swap(::yampi::access& lhs, ::yampi::access& rhs) BOOST_NOEXCEPT
  { lhs.swap(rhs); }
}


# undef YAMPI_MODE
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef YAMPI_addressof

#endif

