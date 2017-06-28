#ifndef YAMPI_VERSION_HPP
# define YAMPI_VERSION_HPP

# include <boost/config.hpp>

# include <mpi.h>

# include <yampi/error.hpp>


namespace yampi
{
  class version_t
  {
    int major_;
    int minor_;

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    BOOST_CONSTEXPR version_t() BOOST_NOEXCEPT_OR_NOTHROW = default;
    version_t(version_t const&) = default;
    version_t& operator=(version_t const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    version_t(version_t&&) = default;
    version_t& operator=(version_t&&) = default;
#   endif
    ~version_t() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR version_t(int const major, int const minor) BOOST_NOEXCEPT_OR_NOTHROW
      : major_(major), minor_(minor)
    { }

    int major() const { return major_; }
    int minor() const { return minor_; }
  };

  inline ::yampi::version_t version()
  {
    int major, minor;
    int const error_code = MPI_Get_version(&major, &minor);
    if (error_code != MPI_SUCCESS)
      throw ::yampi::error(error_code, "yampi::version");

    return ::yampi::version_t(major, minor);
  }
}


#endif

