#ifndef YAMPI_TAG_HPP
# define YAMPI_TAG_HPP

# include <boost/config.hpp>

# include <string>
# include <stdexcept>

# include <mpi.h>


namespace yampi
{
  struct mpi_tag_upper_bound_t { };
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

    explicit tag(::yampi::mpi_tag_upper_bound_t const)
      : mpi_tag_(inquire_upper_bound())
    { }

# ifndef BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
    tag(tag const&) = default;
    tag& operator=(tag const&) = default;
#   ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
    tag(tag&&) = default;
    tag& operator=(tag&&) = default;
#   endif
    ~tag() BOOST_NOEXCEPT_OR_NOTHROW = default;
# endif

    tag& operator++() { ++mpi_tag_; return *this; }
    tag& operator++(int) { mpi_tag_++; return *this; }
    tag& operator--() { --mpi_tag_; return *this; }
    tag& operator--(int) { mpi_tag_--; return *this; }
    tag& operator+=(tag const other) { mpi_tag_ += other.mpi_tag_; return *this; }
    tag& operator-=(tag const other) { mpi_tag_ -= other.mpi_tag_; return *this; }
    tag& operator*=(tag const other) { mpi_tag_ *= other.mpi_tag_; return *this; }
    tag& operator/=(tag const other) { mpi_tag_ /= other.mpi_tag_; return *this; }

    bool operator==(tag const other) const { return mpi_tag_ == other.mpi_tag_; }
    bool operator<(tag const other) const { return mpi_tag_ < other.mpi_tag_; }

    int const& mpi_tag() const { return mpi_tag_; }

   private:
    int inquire_upper_bound() const
    {
      // don't check flag because users cannnot delete the attribute MPI_TAG_UB
      int result, flag;
      int const error_code
        = MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &result, &flag);
      if (error_code != MPI_SUCCESS)
        throw ::yampi::error(error_code, "yampi::tag::inquire_upper_bound");

      return result;
    }
  };

  inline bool operator!=(::yampi::tag const lhs, ::yampi::tag const rhs)
  { return !(lhs == rhs); }

  inline bool operator>=(::yampi::tag const lhs, ::yampi::tag const rhs)
  { return !(lhs < rhs); }

  inline bool operator>(::yampi::tag const lhs, ::yampi::tag const rhs)
  { return rhs < lhs; }

  inline bool operator<=(::yampi::tag const lhs, ::yampi::tag const rhs)
  { return !(rhs < lhs); }

  inline ::yampi::tag operator+(::yampi::tag lhs, ::yampi::tag const rhs)
  { lhs += rhs; return lhs; }

  inline ::yampi::tag operator-(::yampi::tag lhs, ::yampi::tag const rhs)
  { lhs -= rhs; return lhs; }

  inline ::yampi::tag operator*(::yampi::tag lhs, ::yampi::tag const rhs)
  { lhs *= rhs; return lhs; }

  inline ::yampi::tag operator/(::yampi::tag lhs, ::yampi::tag const rhs)
  { lhs /= rhs; return lhs; }

  inline bool is_in_valid_range(::yampi::tag const self)
  {
    ::yampi::tag const mpi_tag_lower_bound = ::yampi::tag(0);
    ::yampi::tag const mpi_tag_upper_bound = ::yampi::tag(::yampi::mpi_tag_upper_bound_t());

    return self >= mpi_tag_lower_bound and self <= mpi_tag_upper_bound;
  }

  ::yampi::tag BOOST_CONSTEXPR_OR_CONST any_tag = ::yampi::tag(::yampi::any_tag_t());
}


#endif

