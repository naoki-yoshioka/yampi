#ifndef YAMPI_TAG_HPP
# define YAMPI_TAG_HPP

# include <cassert>
# include <string>
# include <utility>
# include <type_traits>
# if __cplusplus < 201703L
#   include <boost/type_traits/is_nothrow_swappable.hpp>
# endif
# include <stdexcept>

# include <mpi.h>

# include <yampi/environment.hpp>

# if __cplusplus >= 201703L
#   define YAMPI_is_nothrow_swappable std::is_nothrow_swappable
# else
#   define YAMPI_is_nothrow_swappable boost::is_nothrow_swappable
# endif


namespace yampi
{
  struct tag_upper_bound_t { };
  struct any_tag_t { };

  namespace tags
  {
# if __cplusplus >= 201703L
    inline constexpr ::yampi::tag_upper_bound_t tag_upper_bound{};
    inline constexpr ::yampi::any_tag_t any_tag{};
# else
    constexpr ::yampi::tag_upper_bound_t tag_upper_bound{};
    constexpr ::yampi::any_tag_t any_tag{};
# endif
  }

  class tag
  {
    int mpi_tag_;

   public:
    constexpr tag() noexcept : mpi_tag_(0) { }

    explicit constexpr tag(int const mpi_tag) noexcept
      : mpi_tag_(mpi_tag)
    { }

    explicit constexpr tag(::yampi::any_tag_t const) noexcept
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
    tag(tag const&) = default;
    tag& operator=(tag const&) = default;
    tag(tag&&) = default;
    tag& operator=(tag&&) = default;
    ~tag() noexcept = default;

    constexpr bool operator==(tag const& other) const noexcept
    { return mpi_tag_ == other.mpi_tag_; }

    bool operator<(tag const& other) const noexcept
    {
      assert(mpi_tag_ >= 0);
      assert(other.mpi_tag_ >= 0);
      return mpi_tag_ < other.mpi_tag_;
    }

    tag& operator++() noexcept
    {
      assert(mpi_tag_ >= 0);
      ++mpi_tag_;
      return *this;
    }

    tag& operator--() noexcept
    {
      assert(mpi_tag_ >= 0);
      --mpi_tag_;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      tag&>::type
    operator+=(Integer const n) noexcept
    {
      assert(mpi_tag_ >= 0);
      mpi_tag_ += n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      tag&>::type
    operator-=(Integer const n) noexcept
    {
      assert(mpi_tag_ >= 0);
      mpi_tag_ -= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      tag&>::type
    operator*=(Integer const n) noexcept
    {
      assert(mpi_tag_ >= 0);
      assert(n >= static_cast<Integer>(0));
      mpi_tag_ *= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      tag&>::type
    operator/=(Integer const n) noexcept
    {
      assert(mpi_tag_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_tag_ /= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    template <typename Integer>
    typename std::enable_if<
      std::is_integral<Integer>::value,
      tag&>::type
    operator%=(Integer const n) noexcept
    {
      assert(mpi_tag_ >= 0);
      assert(n > static_cast<Integer>(0));
      mpi_tag_ %= n;
      assert(mpi_tag_ >= 0);
      return *this;
    }

    int operator-(tag const other) const noexcept
    {
      assert(mpi_tag_ >= 0);
      assert(other.mpi_tag_ >= 0);
      return mpi_tag_-other.mpi_tag_;
    }

    constexpr int const& mpi_tag() const noexcept { return mpi_tag_; }
    explicit constexpr operator int() const noexcept { return mpi_tag_; }

    void swap(tag& other) noexcept(YAMPI_is_nothrow_swappable<int>::value)
    {
      using std::swap;
      swap(mpi_tag_, other.mpi_tag_);
    }
  };

  inline constexpr bool operator!=(::yampi::tag const& lhs, ::yampi::tag const& rhs) noexcept
  { return not (lhs == rhs); }

  inline bool operator>=(::yampi::tag const& lhs, ::yampi::tag const& rhs) noexcept
  { return not (lhs < rhs); }

  inline bool operator>(::yampi::tag const& lhs, ::yampi::tag const& rhs) noexcept
  { return rhs < lhs; }

  inline bool operator<=(::yampi::tag const& lhs, ::yampi::tag const& rhs) noexcept
  { return not (rhs < lhs); }

  inline ::yampi::tag operator++(::yampi::tag& lhs, int) noexcept
  { ::yampi::tag result = lhs; ++lhs; return result; }

  inline ::yampi::tag operator--(::yampi::tag& lhs, int) noexcept
  { ::yampi::tag result = lhs; --lhs; return result; }

  template <typename Integer>
  inline ::yampi::tag operator+(::yampi::tag lhs, Integer const rhs) noexcept
  { lhs += rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator-(::yampi::tag lhs, Integer const rhs) noexcept
  { lhs -= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator*(::yampi::tag lhs, Integer const rhs) noexcept
  { lhs *= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator/(::yampi::tag lhs, Integer const rhs) noexcept
  { lhs /= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator%(::yampi::tag lhs, Integer const rhs) noexcept
  { lhs %= rhs; return lhs; }

  template <typename Integer>
  inline ::yampi::tag operator+(Integer const lhs, ::yampi::tag const& rhs) noexcept
  { return rhs+lhs; }

  template <typename Integer>
  inline ::yampi::tag operator*(Integer const lhs, ::yampi::tag const& rhs) noexcept
  { return rhs*lhs; }

  inline void swap(::yampi::tag& lhs, ::yampi::tag& rhs) noexcept(noexcept(lhs.swap(rhs)))
  { lhs.swap(rhs); }


  inline ::yampi::tag tag_upper_bound(::yampi::environment const& environment)
  { return ::yampi::tag(::yampi::tag_upper_bound_t(), environment); }

  inline bool is_valid_tag(::yampi::tag const& self, ::yampi::environment const& environment)
  {
    constexpr ::yampi::tag tag_lower_bound(0);
    return self >= tag_lower_bound and self <= ::yampi::tag_upper_bound(environment);
  }

# if __cplusplus >= 201703L
  inline constexpr ::yampi::tag any_tag{::yampi::tags::any_tag};
# else
  constexpr ::yampi::tag any_tag{::yampi::tags::any_tag};
# endif
}


# undef YAMPI_is_nothrow_swappable

#endif

