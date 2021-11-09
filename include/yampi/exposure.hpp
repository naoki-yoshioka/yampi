#ifndef YAMPI_EXPOSURE_HPP
# define YAMPI_EXPOSURE_HPP

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

# include <yampi/window_base.hpp>
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

  template <typename Window>
  class exposure_guard
  {
    ::yampi::window_base<Window>& window_;

   public:
# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    exposure_guard(exposure_guard const&) = delete;
    exposure_guard& operator=(exposure_guard const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    exposure_guard(exposure_guard const&);
    exposure_guard& operator=(exposure_guard const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

    ~exposure_guard() BOOST_NOEXCEPT
    { MPI_Win_wait(window_.mpi_win()); }

    exposure_guard(::yampi::group const& group, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_(window)
    { do_post(group, 0, environment); }

    exposure_guard(
      ::yampi::group const& group, YAMPI_MODE const assertion, ::yampi::window_base<Window>& window,
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
    unique_exposure() BOOST_NOEXCEPT
      : window_ptr_(nullptr), owns_(false)
    { }

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    unique_exposure(unique_exposure const&) = delete;
    unique_exposure& operator=(unique_exposure const&) = delete;
# else // BOOST_NO_CXX11_DELETED_FUNCTIONS
   private:
    unique_exposure(unique_exposure const&);
    unique_exposure& operator=(unique_exposure const&);

   public:
# endif // BOOST_NO_CXX11_DELETED_FUNCTIONS

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    unique_exposure(unique_exposure&& other)
      : window_ptr_(std::move(other.window_ptr_)), owns_(std::move(other.owns_))
    { other.window_ptr_ = nullptr; other.owns_ = false; }

    unique_exposure& operator=(unique_exposure&& other)
    {
      if (this != YAMPI_addressof(other))
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
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES

    ~unique_exposure() BOOST_NOEXCEPT
    {
      if (owns_)
        MPI_Win_wait(window_ptr_->mpi_win());
    }

    unique_exposure(
      ::yampi::group const& group, ::yampi::window_base<Window>& window, ::yampi::environment const& environment)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { post(group, environment); }

    unique_exposure(
      ::yampi::group const& group, YAMPI_MODE const assertion, ::yampi::window_base<Window>& window,
      ::yampi::environment const& environment)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { post(group, assertion, environment); }

    unique_exposure(::yampi::window_base<Window>& window, ::yampi::defer_exposure_t const)
      : window_ptr_(YAMPI_addressof(window)), owns_(false)
    { }

    unique_exposure(::yampi::window_base<Window>& window, ::yampi::adopt_exposure_t const)
      : window_ptr_(YAMPI_addressof(window)), owns_(true)
    { }

    void post(::yampi::group const& group, ::yampi::environment const& environment) const
    { do_post(group, 0, environment); }

    void post(::yampi::group const& group, YAMPI_MODE const assertion, ::yampi::environment const& environment) const
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
      int const error_code = MPI_Win_test(window_ptr_->mpi_win(), YAMPI_addressof(result));
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::exposure::test", environment);

      if (static_cast<bool>(result))
        owns_ = false;

      return static_cast<bool>(result);
    }

    void swap(exposure& other) BOOST_NOEXCEPT
    {
      using std::swap;
      swap(window_ptr_, other.window_ptr_);
      swap(owns_, other.owns_);
    }

    ::yampi::window_base<Window>* window_ptr() const BOOST_NOEXCEPT_OR_NOTHROW { return window_ptr_; }
    bool owns_exposure() const BOOST_NOEXCEPT_OR_NOTHROW { return owns_; }
  };

  template <typename Window>
  inline void swap(::yampi::exposure<Window>& lhs, ::yampi::exposure<Window>& rhs) BOOST_NOEXCEPT
  { lhs.swap(rhs); }
}


# undef YAMPI_MODE
# ifdef BOOST_NO_CXX11_NULLPTR
#   undef nullptr
# endif
# undef YAMPI_addressof

#endif

