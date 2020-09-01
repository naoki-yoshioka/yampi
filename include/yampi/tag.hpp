#ifndef YAMPI_TAG_HPP
# define YAMPI_TAG_HPP

# include <boost/config.hpp>

# include <cassert>
# include <string>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
#   if __cplusplus < 201703L
#     include <boost/type_traits/is_nothrow_swappable.hpp>
#   endif
# else
#   include <boost/utility/enable_if.hpp>
#   include <boost/type_traits/is_integral.hpp>
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <stdexcept>

# include <mpi.h>

# include <yampi/environment.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define YAMPI_enable_if std::enable_if
#   define YAMPI_is_integral std::is_integral
# else
#   define YAMPI_enable_if boost::enable_if_c
#   define YAMPI_is_integral boost::is_integral
# endif

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct tag_upper_bound_t { };
  struct any_tag_t { };

  class tag
  {
    int mpi_tag_;

   public:
    BOOST_CONSTEXPR tag() BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_tag_(0)
    { }

    explicit BOOST_CONSTEXPR tag(int const mpi_tag) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_tag_(mpi_tag)
    { }

    explicit BOOST_CONSTEXPR tag(::yampi::any_tag_t const) BOOST_NOEXCEPT_OR_NOTHROW
      : mpi_tag_(MPI_ANY_TAG)
    { }

    tag(::yampi::tag_upper_bound_t const, ::yampi::environment const& environment)
      : mpi_tag_(inquire_upper_bound(environment))
    { }

   private:
    int inquire_upper_bound(::yampi::environment const& environment) const
    {
      // don't check flag because users cannnot delete the attribute MPI_TAG_UB
      int result, flag;
      int const error_code
        = MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &result, &flag);

      return error_code == MPI_SUCCESS
        ? result
        : throw ::yampi::error(error_code, "yampi::tag::inquire_upper_bound", environment);
    }

   public:
# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    tag(tag const&) = default;
    tag& operator=(tag const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    tag(tag&&) = default;
    tag& operator=(tag&&) = default;
#   endif
    ~tag() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    BOOST_CONSTEXPR bool operator==(tag const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    { return mpi_tag_ == other.mpi_tag_; }

    bool operator<(tag const& other) const BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      assert(other.mpi_tag_ >= 0);
      return mpi_tag_ < other.mpi_tag_;
    }

    tag& operator++() BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      ++mpi_tag_;
      return *this;
    }

    tag& operator--() BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      --mpi_tag_;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      tag&>::type
    operator+=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      mpi_tag_ += n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      tag&>::type
    operator-=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      mpi_tag_ -= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      tag&>::type
    operator*=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_tag_ *= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      tag&>::type
    operator/=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_tag_ /= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename YAMPI_enable_if<
      YAMPI_is_integral<Integer>::value,
      tag&>::type
    operator%=(Integer const n) BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_tag_ %= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    int operator-(tag const other) const BOOST_NOEXCEPT_OR_NOTHROW
    {
      assert(mpi_tag_ >= 0);
      assert(other.mpi_tag_ >= 0);
      return mpi_tag_-other.mpi_tag_;
    }

    BOOST_CONSTEXPR int const& mpi_tag() const BOOST_NOEXCEPT_OR_NOTHROW { return mpi_tag_; }
# ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
    explicit BOOST_CONSTEXPR operator int() const { return mpi_tag_; }
# else
    BOOST_CONSTEXPR operator int() const { return mpi_tag_; }
# endif

    void swap(tag& other)
      BOOST_NOEXCEPT_IF(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_tag_, other.mpi_tag_);
    }
  };

  inline BOOST_CONSTEXPR bool operator!=(::yampi::tag const& lhs, ::yampi::tag const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::tag const& lhs, ::yampi::tag const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::tag const& lhs, ::yampi::tag const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs < lhs; }

  inline bool operator<=(::yampi::tag const& lhs, ::yampi::tag const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return not (rhs < lhs); }

  inline ::yampi::tag operator++(::yampi::tag& lhs, int)
    BOOST_NOEXCEPT_OR_NOTHROW
  { ::yampi::tag result = lhs; ++lhs; return result; }

  inline ::yampi::tag operator--(::yampi::tag& lhs, int)
    BOOST_NOEXCEPT_OR_NOTHROW
  { ::yampi::tag result = lhs; --lhs; return result; }

  template <typename Integer>
  inline ::yampi::tag operator+(::yampi::tag lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs += rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator-(::yampi::tag lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator*(::yampi::tag lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator/(::yampi::tag lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator%(::yampi::tag lhs, Integer const rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator+(Integer const lhs, ::yampi::tag const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs+lhs; }

  template <typename Integer>
  inline ::yampi::tag operator*(Integer const lhs, ::yampi::tag const& rhs)
    BOOST_NOEXCEPT_OR_NOTHROW
  { return rhs*lhs; }

  inline void swap(::yampi::tag& lhs, ::yampi::tag& rhs)
    BOOST_NOEXCEPT_IF(BOOST_NOEXCEPT_EXPR(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  inline BOOST_CONSTEXPR ::yampi::tag any_tag() BOOST_NOEXCEPT_OR_NOTHROW
  { return tag(::yampi::any_tag_t()); }

  inline ::yampi::tag tag_upper_bound(::yampi::environment const& environment)
  { return ::yampi::tag(::yampi::tag_upper_bound_t(), environment); }

  inline bool is_valid_tag(::yampi::tag const& self, ::yampi::environment const& environment)
  {
    BOOST_CONSTEXPR_OR_CONST ::yampi::tag tag_lower_bound(0);
    return self >= tag_lower_bound and self <= ::yampi::tag_upper_bound(environment);
  }
}


# undef YAMPI_is_nothrow_swappable
# undef YAMPI_is_integral
# undef YAMPI_enable_if

#endif

