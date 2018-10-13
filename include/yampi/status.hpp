#ifndef YAMPI_STATUS_HPP
# define YAMPI_STATUS_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>
# include <yampi/utility/is_nothrow_swappable.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_enable_if boost::enable_if_c
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define YAMPI_addressof std::addressof
# else
#   define YAMPI_addressof boost::addressof
# endif

# if MPI_VERSION >= 3
#   define YAMPI_Get_elements MPI_Get_elements_x
# else
#   define YAMPI_Get_elements MPI_Get_elements
# endif


namespace yampi
{
  class count_value_undefined_error
    : public std::runtime_error
  {
   public:
    count_value_undefined_error()
      : std::runtime_error("count in MPI_GET_COUNT is MPI_UNDEFINED")
    { }
  };

  class status
  {
    MPI_Status mpi_status_;

   public:
# if MPI_VERSION >= 3
    typedef MPI_Count num_elements_type;
# else
    typedef int num_elements_type;
# endif

# ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
    status() = delete;
# else
   private:
    status();

   public:
# endif

    explicit status(MPI_Status const& stat) BOOST_NOEXCEPT_OR_NOTHROW : mpi_status_(stat) { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    status(status const&) = default;
    status& operator=(status const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    status(status&&) = default;
    status& operator=(status&&) = default;
#   endif
    ~status() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    ::yampi::rank source() const BOOST_NOEXCEPT_OR_NOTHROW
    { return ::yampi::rank(mpi_status_.MPI_SOURCE); }
    ::yampi::tag tag() const BOOST_NOEXCEPT_OR_NOTHROW
    { return ::yampi::tag(mpi_status_.MPI_TAG); }

    void test_error(::yampi::environment const& environment) const
    {
      if (mpi_status_.MPI_ERROR != MPI_SUCCESS)
        throw ::yampi::error(mpi_status_.MPI_ERROR, "yampi::status::test_error", environment);
    }

    int message_length(
      ::yampi::datatype const& datatype, ::yampi::environment const& environment) const
    {
      int count;
      int const error_code
        = MPI_Get_count(
            const_cast<MPI_Status*>(YAMPI_addressof(mpi_status_)),
            datatype.mpi_datatype(), &count);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::status::message_length", environment);
      if (count == MPI_UNDEFINED)
        throw ::yampi::count_value_undefined_error();

      return count;
    }

    num_elements_type num_elements(
      ::yampi::datatype const& datatype, ::yampi::environment const& environment) const
    {
      num_elements_type result;
      int const error_code
        = YAMPI_Get_elements(
            const_cast<MPI_Status*>(YAMPI_addressof(mpi_status_)),
            datatype.mpi_datatype(), &result);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::status::num_elements", environment);

      return result;
    }

    bool empty() const
    {
      return mpi_status_.MPI_TAG == MPI_ANY_TAG
        and mpi_status_.MPI_SOURCE == MPI_ANY_SOURCE
        and mpi_status_.MPI_ERROR == MPI_SUCCESS;
    }

    void swap(status& other)
      BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable<MPI_Status>::value)
    {
      using std::swap;
      swap(mpi_status_, other.mpi_status_);
    }
  };

  inline void swap(::yampi::status& lhs, ::yampi::status& rhs)
    BOOST_NOEXCEPT_IF(::yampi::utility::is_nothrow_swappable< ::yampi::status >::value)
  { lhs.swap(rhs); }


  class ignore_status_t { };

  inline BOOST_CONSTEXPR ::yampi::ignore_status_t ignore_status()
  { return ::yampi::ignore_status_t(); }
}


# undef YAMPI_addressof
# undef YAMPI_enable_if

#endif

