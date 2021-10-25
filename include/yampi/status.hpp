#ifndef YAMPI_STATUS_HPP
# define YAMPI_STATUS_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/type_traits/has_nothrow_copy.hpp>
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
# endif

# include <mpi.h>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/rank.hpp>
# include <yampi/tag.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_is_nothrow_copy_constructible std::is_nothrow_copy_constructible
#   define YAMPI_enable_if std::enable_if
# else
#   define YAMPI_is_nothrow_copy_constructible boost::has_nothrow_copy_constructor
#   define YAMPI_enable_if boost::enable_if_c
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
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

    explicit status(MPI_Status const& stat)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible<MPI_Status>::value)
      : mpi_status_(stat)
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    status(status const&) = default;
    status& operator=(status const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    status(status&&) = default;
    status& operator=(status&&) = default;
#   endif
    ~status() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    ::yampi::rank source() const
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::rank >::value)
    { return ::yampi::rank(mpi_status_.MPI_SOURCE); }

    ::yampi::tag tag() const
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_copy_constructible< ::yampi::tag >::value)
    { return ::yampi::tag(mpi_status_.MPI_TAG); }

    void test_error(::yampi::environment const& environment) const
    {
      if (mpi_status_.MPI_ERROR != MPI_SUCCESS)
        throw ::yampi::error(mpi_status_.MPI_ERROR, "yampi::status::test_error", environment);
    }

    template <typename Datatype>
    int message_length(
      ::yampi::datatype_base<Datatype> const& datatype, ::yampi::environment const& environment) const
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

    template <typename Datatype>
    num_elements_type num_basic_elements(
      ::yampi::datatype_base<Datatype> const& datatype, ::yampi::environment const& environment) const
    {
      num_elements_type result;
      int const error_code
        = YAMPI_Get_elements(
            const_cast<MPI_Status*>(YAMPI_addressof(mpi_status_)),
            datatype.mpi_datatype(), &result);

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::status::num_basic_elements", environment);
    }

    bool empty() const
      BOOST_NOEXCEPT_IF(
        BOOST_NOEXCEPT_EXPR(mpi_status_.MPI_TAG == MPI_ANY_TAG)
        and BOOST_NOEXCEPT_EXPR(mpi_status_.MPI_SOURCE == MPI_ANY_SOURCE)
        and BOOST_NOEXCEPT_EXPR(mpi_status_.MPI_ERROR == MPI_SUCCESS))
    {
      return mpi_status_.MPI_TAG == MPI_ANY_TAG
        and mpi_status_.MPI_SOURCE == MPI_ANY_SOURCE
        and mpi_status_.MPI_ERROR == MPI_SUCCESS;
    }

    MPI_Status const& mpi_status() const { return mpi_status_; }

    void swap(status& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<MPI_Status>::value)
    {
      using std::swap;
      swap(mpi_status_, other.mpi_status_);
    }
  };

  inline void swap(::yampi::status& lhs, ::yampi::status& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  class ignore_status_t { };

  inline BOOST_CONSTEXPR ::yampi::ignore_status_t ignore_status() BOOST_NOEXCEPT_OR_NOTHROW
  { return ::yampi::ignore_status_t(); }
}


# undef YAMPI_addressof
# undef YAMPI_is_nothrow_swappable
# undef YAMPI_enable_if
# undef YAMPI_is_nothrow_copy_constructible

#endif

