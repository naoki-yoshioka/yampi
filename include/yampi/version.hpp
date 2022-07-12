#ifndef YAMPI_VERSION_HPP
# define YAMPI_VERSION_HPP

# if __cplusplus >= 201703L
#   include <type_traits>
# else
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif

# include <mpi.h>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  class version_error
    : public std::runtime_error
  {
    int error_code_;

   public:
    explicit version_error(int const error_code)
     : std::runtime_error("Error occurred when getting version"),
       error_code_(error_code)
    { }

    int error_code() const { return error_code_; }
  };


  class version_t
  {
    int major_;
    int minor_;

   public:
    constexpr version_t() noexcept = default;
    version_t(version_t const&) = default;
    version_t& operator=(version_t const&) = default;
    version_t(version_t&&) = default;
    version_t& operator=(version_t&&) = default;
    ~version_t() noexcept = default;

    constexpr version_t(int const major, int const minor) noexcept
      : major_(major), minor_(minor)
    { }

    int major() const noexcept { return major_; }
    int minor() const noexcept { return minor_; }

    void swap(version_t& other) noexcept
    {
      using std::swap;
      swap(major_, other.major_);
      swap(minor_, other.minor_);
    }
  };

  inline bool operator==(::yampi::version_t const& lhs, ::yampi::version_t const& rhs) noexcept
  { return lhs.major() == rhs.major() and lhs.minor() == rhs.minor(); }

  inline bool operator!=(::yampi::version_t const& lhs, ::yampi::version_t const& rhs) noexcept
  { return not (lhs == rhs); }

  inline bool operator<(::yampi::version_t const& lhs, ::yampi::version_t const& rhs) noexcept
  {
    return lhs.major() < rhs.major()
      or (lhs.major() == rhs.major() and lhs.minor() < rhs.minor());
  }

  inline bool operator>=(::yampi::version_t const& lhs, ::yampi::version_t const& rhs) noexcept
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::version_t const& lhs, ::yampi::version_t const& rhs) noexcept
  { return rhs < lhs; }

  inline bool operator<=(::yampi::version_t const& lhs, ::yampi::version_t const& rhs) noexcept
  { return not (rhs < lhs); }

  inline void swap(::yampi::version_t& lhs, ::yampi::version_t& rhs) noexcept
  { lhs.swap(rhs); }


  inline ::yampi::version_t version()
  {
    int major, minor;
    int const error_code = MPI_Get_version(&major, &minor);

    return error_code == MPI_SUCCESS
      ? ::yampi::version_t(major, minor)
      : throw ::yampi::version_error(error_code);
  }
}


# undef YAMPI_is_nothrow_swappable

#endif

