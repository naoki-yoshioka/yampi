#if MPI_VERSION >= 3
# ifndef YAMPI_SPLIT_TYPE_HPP
#   define YAMPI_SPLIT_TYPE_HPP

#   include <boost/config.hpp>

#   include <cassert>
#   include <string>
#   include <utility>
#   ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#     include <type_traits>
#     if __cplusplus < 201703L
#       include <boost/type_traits/is_nothrow_swappable.hpp>
#     endif
#   else
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
#   include <stdexcept>

#   include <mpi.h>

#   include <yampi/environment.hpp>

#   if __cplusplus >= 201703L
#     define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
#   else
#     define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
#   endif


namespace yampi
{
  struct shared_memory_split_type_t { };
  struct undefined_split_type_t { };

  class split_type
  {
    int mpi_split_type_;

   public:
    BOOST_CONSTEXPR split_type() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_split_type_(0)
    { }

    explicit BOOST_CONSTEXPR split_type(int const mpi_split_type) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_split_type_(mpi_split_type)
    { }

    explicit BOOST_CONSTEXPR split_type(::yampi::shared_memory_split_type_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_split_type_(MPI_COMM_TYPE_SHARED)
    { }

    explicit BOOST_CONSTEXPR split_type(::yampi::undefined_split_type_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_split_type_(MPI_UNDEFINED)
    { }

#   ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    split_type(split_type const&) = default;
    split_type& operator=(split_type const&) = default;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    split_type(split_type&&) = default;
    split_type& operator=(split_type&&) = default;
#     endif
    ~split_type() BOOST_NOEXCEPT_OR_NOTHROW = default;
#   endif

    BOOST_CONSTEXPR bool operator==(split_type const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_split_type_ == other.mpi_split_type_; }

    BOOST_CONSTEXPR int const& mpi_split_type() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_split_type_; }

    void swap(split_type& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_split_type_, other.mpi_split_type_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::split_type const& lhs, ::yampi::split_type const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline void swap(::yampi::split_type& lhs, ::yampi::split_type& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }
}


#   undef YAMPI_is_nothrow_swappable

# endif
#endif

